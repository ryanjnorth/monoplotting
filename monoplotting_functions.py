"""
monoplotting_functions.py — engine for the Historical Photo Monoplotting notebook.

Every processing step of the notebook lives here as a function. The notebook
holds the configuration blocks and calls these functions in order; each
function takes the values it needs as inputs and returns the values later
cells need, so nothing here reads the notebook's variables directly.

Functions are grouped by the notebook cell they belong to:

    Part 1 — Camera model and raycasting      Cells 1–9
    Part 2 — Line feature projection          Cells 10–13
    Part 3 — Orthorectification (optional)    Cells 14–15

Designed by Dr Ryan North and Elena Disilvestro, Monash University.

Install:
    pip install numpy opencv-python rasterio open3d matplotlib scipy geopandas shapely

Note, Python 3.12 is required for Open3D to work.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Cell 1 — Imports
# ══════════════════════════════════════════════════════════════════════════════

# ── Core scientific stack ─────────────────────────────────────────────────────
import numpy as np                          # all array maths
import matplotlib.pyplot as plt             # all plots
import matplotlib.tri as mtri               # triangulation plotting (mesh cell)
import matplotlib.colors as mcolors         # colourmap normalisation (mesh cell)
import os, csv, warnings                    # paths, CSV reading, warning control
from collections import OrderedDict         # keeps features in digitising order

# ── Image and raster I/O ──────────────────────────────────────────────────────
import cv2                                  # image loading + camera geometry
import rasterio                             # GeoTIFF read/write
from rasterio.transform import from_bounds  # building an output grid on resample
import rasterio.warp                        # resampling onto another grid (only used on the resample branches)

# ── Interpolation ─────────────────────────────────────────────────────────────
from scipy.interpolate import RBFInterpolator  # thin plate spline warp (Cell 15)

# ── Vector output (optional dependency) ───────────────────────────────────────
# Only needed to write the shapefile in Cell 12.
try:
    import geopandas as gpd                                   # geopandas: tables whose rows carry a geometry
    from shapely.geometry import LineString, MultiLineString  # two geometry types: one line, and a group of lines
    GEOPANDAS_OK = True
except ImportError:
    GEOPANDAS_OK = False
    print('geopandas/shapely not installed — shapefile output (Cell 12) will not work.')
    print('  Install with: pip install geopandas shapely')

# ── 3D meshing and raycasting (required for Cells 8, 9, 11) ──────────────────
# Open3D does the heavy lifting: it builds a 'bounding volume heirarchy' (a tree of nested boxes that lets a ray skip most of the mesh)
# over the terrain triangles so millions of ray-triangle intersection tests run in seconds.
try:
    import open3d as o3d                       # Open3D: 3D meshes and fast ray intersection
    OPEN3D_OK = True
except ImportError:
    OPEN3D_OK = False
    print('open3d not installed — mesh generation and raycasting will not work.')
    print('  Install with: pip install open3d')


def report_versions():
    """Print the versions of the main libraries."""
    # ── Report versions so a broken environment is obvious immediately ────────────
    print('Imports OK')
    print(f'  numpy     {np.__version__}')
    print(f'  opencv    {cv2.__version__}')
    print(f'  rasterio  {rasterio.__version__}')
    if OPEN3D_OK:    print(f'  open3d    {o3d.__version__}')
    if GEOPANDAS_OK: print(f'  geopandas {gpd.__version__}')


# ══════════════════════════════════════════════════════════════════════════════
# Cell 2 — File paths
# ══════════════════════════════════════════════════════════════════════════════

def check_paths(image_path, dem_path, gcp_csv, mesh_path, output_path,
                features_csv, output_shp, custom_mask_path, stable_mask_path):
    """Print the file paths set in Cell 2 and warn about missing required inputs."""
    # ── Report ────────────────────────────────────────────────────────────────────
    print('File paths set.')
    print(f'  Image        : {image_path}')
    print(f'  DEM          : {dem_path}')
    print(f'  GCPs         : {gcp_csv}')
    print(f'  Mesh         : {mesh_path}')
    print(f'  Ortho out    : {output_path}')
    print(f'  Features     : {features_csv}')
    print(f'  Shapefile out: {output_shp}')
    print(f'  Custom mask  : {custom_mask_path}')
    print(f'  Stable mask  : {stable_mask_path}')

    # Check paths/names exist now rather than in middle of a cell later.
    for label, path in (('IMAGE_PATH', image_path),
                        ('DEM_PATH',   dem_path),
                        ('GCP_CSV',    gcp_csv)):
        if not os.path.exists(path):
            print(f'\n  WARNING: {label} does not exist: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Cell 3 — Load photo and GCPs
# ══════════════════════════════════════════════════════════════════════════════

def load_photo(image_path):
    """
    Load the oblique photo.

    Returns
    -------
    img_cv  : image in its original channel order (BGR for colour), used for SAMPLING pixels
    img_rgb : RGB copy, used only for display with matplotlib
    h_img, w_img : image height and width in pixels
    aspect  : width / height
    n_bands : 1 for greyscale, 3 for colour
    """
    # ── Load the photo ────────────────────────────────────────────────────────────
    # IMREAD_UNCHANGED keeps the file as it is on disk (greyscale stays greyscale)
    # instead of silently forcing it to be three channels/RGB.
    img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)               # read the file into an array of pixel values
    if img_cv is None:                                                  # OpenCV returns None instead of raising on failure
        raise FileNotFoundError(f'Cannot load image: {image_path}')     # "raise" stops the cell with an error message

    # .shape is a tuple of the array's dimensions; [:2] takes the first two of them,
    # and the two names on the left receive them in order.
    h_img, w_img = img_cv.shape[:2]   # image height and width in pixels
    aspect = w_img / h_img            # used in Cell 4 to derive sensor height

    # Two arrays are kept from here on:
    #   img_cv  — original channel order (BGR for colour), used for SAMPLING pixels
    #   img_rgb — RGB copy, used only for display with matplotlib
    if img_cv.ndim == 2:                                            # .ndim is the number of dimensions. 2 dimensions is row and col (no value channel for greyscale)
        # Greyscale.
        n_bands = 1                                                 # single band for greyscale
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)          # copy the grey value into R, G and B so matplotlib can show it
        print(f'Image loaded: {w_img} x {h_img} px  (greyscale)')   # report what was loaded
    elif img_cv.shape[2] == 4:
        #RGBA
        img_cv  = img_cv[:, :, :3]                                  # drop the alpha channel, it carries no scene information. Slice to keep all rows, all cols, and first 3 channels (RGB)
        n_bands = 3                                                 # 3 colour bands
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)           # Convert colour profile. OpenCV stores Blue-Green-Red, but matplotlib wants Red-Green-Blue
        print(f'Image loaded: {w_img} x {h_img} px  (RGBA — alpha stripped)') # report the size and that the alpha channel was dropped
    else:
        #RGB
        n_bands = 3                                                 # 3 colour bands
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)           # Convert colour profile. OpenCV stores Blue-Green-Red, but matplotlib wants Red-Green-Blue
        print(f'Image loaded: {w_img} x {h_img} px  (colour)')      # report the size for an ordinary colour photo

    print(f'  Aspect ratio: {aspect:.3f}   Bands: {n_bands}')       #report aspect ratio (to 3 decimal places) and number of bands in loaded photo

    return img_cv, img_rgb, h_img, w_img, aspect, n_bands


# ── Load GCPs from CSV ────────────────────────────────────────────────────────
def load_gcps(csv_path, img_w, img_h):
    """
    Read GCP pixel/world coordinate pairs from CSV.

    Expected header (surrounding spaces are ignored, capitalisation is not):
        GCP, row, col, X, Y, Z
    where row/col are image pixel coordinates and X/Y/Z are Easting/Northing/
    Elevation in the DEM's projected CRS, in metres.

    Returns
    -------
    gcp_pixel : (N, 2) float array of [col, row]  — OpenCV (x, y) order
    gcp_world : (N, 3) float array of [X, Y, Z]
    labels    : list of N label strings
    """
    with open(csv_path, newline='') as f:                           # open csv file as 'f'
        reader = csv.DictReader(f)                                  # reads each line into a dict keyed by the header names
        if reader.fieldnames is None:                               # fail if fieldnames (header) is empty and raise error
            raise ValueError(f'{csv_path} appears to be empty.')     # stop here rather than failing further down

        # Column names must be exactly: GCP, row, col, X, Y, Z
        reader.fieldnames = [h.strip() for h in reader.fieldnames]     # replaces the column names so a header written with spaces after the commas still works
        missing = [c for c in ('GCP', 'row', 'col', 'X', 'Y', 'Z')     # take each name c from the tuple, keep it only if c is not in reader.fieldnames, and collect what survives into a list
                   if c not in reader.fieldnames]
        if missing:                                                     # If header has everything, 'missing' is empty (good!). A if something is missing then error is raised
            raise ValueError(f'GCP CSV is missing columns {missing}. '
                             f'Found: {reader.fieldnames}')

        labels, gcp_pixel, gcp_world = [], [], []                   # create three empty lists

        for rec in reader:                                          # loop over the data rows. Variable named 'rec' short for record, is one row as a dict
            if rec['GCP'] is None or not rec['GCP'].strip():        # missing or blank label
                continue                                            # skip blank lines
            label = rec['GCP'].strip()                              # the label text, trimmed of spaces
            col   = float(rec['col'])                               # float() turns the text "1523" into the number 1523.0
            row   = float(rec['row'])                               # same for the row

            # Generate a warning if GCP is outside the image pixel space
            # A GCP outside the frame is often a row/col swap
            # Python allows chained comparisons: 0 <= col < img_w reads as one test.
            if not (0 <= col < img_w and 0 <= row < img_h):
                print(f'  WARNING: GCP "{label}" at (col={col:.0f}, row={row:.0f}) '   # warn but carry on, in case the point really is off-frame
                      f'is outside the image ({img_w} x {img_h}) — '
                      f'are row and col swapped?')

            labels.append(label)                                    # .append adds one item to the end of a list
            gcp_pixel.append([col, row])                            # [col, row] = OpenCV (x, y)
            gcp_world.append([float(rec['X']),                      # easting
                              float(rec['Y']),                      # northing
                              float(rec['Z'])])                     # elevation

    # Warn if not enough GCPs.
    # solvePnP needs at least 4 GCPs. Certain georectification methods require at least 6. More than 8 gives RANSAC (outlier removal) room to work.
    if len(labels) < 4:                                             # len() is the number of items in a list
        raise ValueError(f'Only {len(labels)} GCPs found — at least 4 are required, min 6 recommended.')


    # Sort by label so the printed tables are stable between runs.
    order = np.argsort(labels)                                      # argsort returns the positions that would sort the list

    # return the results of this function
    # the brackets group three values into one tuple. Indexing an array by a list of positions, (here `order`) reorders it.
    return (np.array(gcp_pixel, dtype=np.float64)[order],           # pixel coordinates as a numpy array, sorted
            np.array(gcp_world, dtype=np.float64)[order],           # world coordinates, sorted the same way
            [labels[i] for i in order])                             # the labels, sorted the same way


def print_gcp_table(gcp_labels, gcp_pixel, gcp_world):
    """Print the loaded GCPs as a table."""
    #print to view the results
    print(f'\n{len(gcp_labels)} GCPs loaded:')                                      # blank line, then the count
    print(f'  {"GCP":<12} {"col":>8} {"row":>8}   {"X":>12} {"Y":>13} {"Z":>8}')    # :<12 left-aligns in a 12-character column, :>8 right-aligns in 8 — this builds a table header.
    print(f'  {"-"*66}')                                                            # "-"*66 repeats the dash 66 times

    for label, px, wc in zip(gcp_labels, gcp_pixel, gcp_world):                     # zip() walks several sequences together, handing one item from each per pass.
        print(f'  {label:<12} {px[0]:>8.1f} {px[1]:>8.1f}   '                       # label, then column and row
              f'{wc[0]:>12.1f} {wc[1]:>13.1f} {wc[2]:>8.1f}')                       # then easting, northing, elevation


def exclude_gcps(gcp_pixel, gcp_world, gcp_labels, exclude):
    """
    Drop the GCPs whose labels are listed in `exclude`.

    Returns gcp_pixel, gcp_world, gcp_labels in the same order as load_gcps.
    """
    # ── Drop excluded GCPs ────────────────────────────────────────────────────────
    if exclude:                                                         # an empty list is False, so this runs only if labels were listed
        keep       = [i for i, l in enumerate(gcp_labels) if l not in exclude]   # keep is a list of i, for each position i and label l in gcp_labels, where l is not in EXCLUDE_GCPS.
        gcp_labels = [gcp_labels[i] for i in keep]                      # rebuild the label list from those positions
        gcp_pixel  = gcp_pixel[keep]                                    # numpy indexes by a list of positions directly
        gcp_world  = gcp_world[keep]                                    # same for the world coordinates
        print(f'\nExcluded: {exclude}')                                 # report what was dropped
        print(f'Remaining ({len(gcp_labels)}): {gcp_labels}')           # and what is left
        if len(gcp_labels) < 4:                                         # the solve needs 4
            raise ValueError('Fewer than 4 GCPs remain after exclusion.')   # too many were excluded to solve the pose
    return gcp_pixel, gcp_world, gcp_labels


def plot_gcps(img_rgb, gcp_labels, gcp_pixel):
    """Draw the GCPs on the photo so their positions can be checked."""
    # ── Plot GCPs on the photo to visually check ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))                             # subplots() returns a figure and its axes; figsize is in inches.
    ax.imshow(img_rgb)                                                  # draw the photo as the background
    for label, px in zip(gcp_labels, gcp_pixel):                        # one marker per GCP
        ax.plot(px[0], px[1], 'r+', markersize=14, markeredgewidth=2)   # 'r+' = a red plus sign at (col, row)
        ax.annotate(label, xy=(px[0], px[1]), xytext=(6, -14),          # write the label near the marker
                    textcoords='offset points', color='red', fontsize=9,   # offset is in points from the marker
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))   # a semi-transparent white box behind the text
    ax.set_title(f'{len(gcp_labels)} GCPs — each cross should sit on its measured feature')   # figure title
    ax.axis('off')                                                      # hide the pixel-coordinate axes
    plt.tight_layout()                                                  # tidy the spacing
    plt.show()                                                          # display plot

    print('\nIf the markers land on the right features, continue to Cell 4.')   # closing instructions to the user
    print('If they look mirrored or transposed, swap row/col in the GCP CSV.')  # and the most common thing to check if GCPs in wrong position


# ══════════════════════════════════════════════════════════════════════════════
# Cell 4 — Resolve camera field of view
# ══════════════════════════════════════════════════════════════════════════════

# Function to convert focal length (mm) and sensor size (mm) to vertical FOV
def focal_and_sensor_to_fov(focal_mm, sensor_height_mm):
    """Vertical FOV in degrees from physical focal length and sensor height."""
    return 2.0 * np.degrees(np.arctan(sensor_height_mm / (2.0 * focal_mm)))

# Function to convert vertical FOV to focal length in pixels
def fov_to_focal_px(fov_deg, image_height_px):
    """Focal length in PIXELS from vertical FOV — this is what goes into intrinsic camera matrix ('K')."""
    return image_height_px / (2.0 * np.tan(np.radians(fov_deg / 2.0)))


#Function to solve likely FOV.
def reproj_error_at_fov(fov_deg, gcp_world_c, gcp_pixel, w_img, h_img, ransac_threshold=20.0):
    """
    Solve the camera pose at one candidate FOV and return a ROBUST reprojection
    error in pixels (1e6 if the solve fails).

    The pose is solved with RANSAC and scored with a capped ('MSAC') error, so one
    badly matched GCP cannot drag the FOV estimate towards itself.

    gcp_world_c : GCP world coordinates with their mean subtracted (see
                  estimate_fov_from_gcps)

    Returns
    -------
    score     : sqrt(mean(min(error^2, threshold^2))), in pixels
    n_inliers : number of GCPs within ransac_threshold of their picked position
    errors    : (N,) per-GCP reprojection error in pixels, or None if the solve failed
    """
    if fov_deg <= 1 or fov_deg >= 179:                                          # geometrically impossible values
        return 1e6, 0, None                                                     # dummy value to signify error

    # calculate focal lenth (px) for this trial FOV
    focal_t = fov_to_focal_px(fov_deg, h_img)

    # Build intrinsic camera matrix ('K') for the trial FOV.
    # np.array() from a list of lists builds a 2D matrix, one inner list per row.
    K_t = np.array([[focal_t, 0,       w_img / 2.0 - 0.5],                      # row 1: focal length and the principal point's x
                    [0,       focal_t, h_img / 2.0 - 0.5],                      # row 2: focal length and the principal point's y (focal length is same in x and y dimensions if pixels are square)
                    [0,       0,       1.0              ]],                     # row 3: fixed, copies z through for the perspective divide (col = f*x/z + cx)
                    dtype=np.float64)                                           # dtype sets 64-bit precision

    # try/except because solvePnP can fail easily
    try:
        # Solve the pose with RANSAC rather than plain least squares.
        # Plain least squares minimises the sum of SQUARED errors, so a GCP that is 50 px
        # off counts 2500 times more than one that is 1 px off, and the pose bends towards
        # it. RANSAC instead finds the pose that most GCPs agree with (within
        # ransac_threshold px), then refines it using only those GCPs - the same approach
        # Cell 5 uses for the final pose.
        # OpenCV's RANSAC uses its own fixed random seed, so repeat runs give the same answer.
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(                           # Four returned values: a success flag, the rotation, the translation, and the GCPs RANSAC trusted.
            gcp_world_c.reshape(-1, 1, 3).astype(np.float64),                   # Provide world coords. reshape to the (N rows, 1 col, 3 values) layout OpenCV expects. -1 means automatically work out how many rows there are
            gcp_pixel.reshape(-1, 1, 2).astype(np.float64),                     # provide pixel coords. Reshape same as above.
            K_t,                                                                # provide camera matrix
            np.array([0.0, 0.0, 0.0, 0.0]),                                     # provide lens distortion coefficients in format (k1, k2, p1, p2) (two radial distortion terms, two tangential distorition terms). Here assuming none, so all zeros.
            iterationsCount=1000,                                               # random subsets to try (it stops early once confident, so this is rarely reached)
            reprojectionError=ransac_threshold,                                 # a GCP further than this from its picked position (px) is an outlier
            confidence=0.999,                                                   # stop early once this confident
            flags=cv2.SOLVEPNP_ITERATIVE)                                       # Iterative solve method is general purpose method. Gets rough pose then refines with until reprojection error stops improving.
        if not ok:
            return 1e6, 0, None                                                 # return error signal if not fails

        # Project the GCPs back through the trial model and measure the miss.
        # The second returned value (Jacobian optimiser values) is not needed, and _ is the conventional name for "ignore this".
        reproj, _ = cv2.projectPoints(
            gcp_world_c.reshape(-1, 1, 3), rvec, tvec,   # the world points and the rotation and translation pose just solved
            K_t, np.array([0.0, 0.0, 0.0, 0.0]))                            # the same intrinsics and no distortion
        errors = np.linalg.norm(gcp_pixel - reproj.reshape(-1, 2), axis=1)      # norm(..., axis=1) gives the straight-line error distance per GCP, in px

        # Score with a capped ('MSAC') error rather than the mean.
        # Each GCP contributes its squared error, but never more than threshold^2. A GCP
        # 50 px off then costs the same as one 8 px off, so a blunder adds the same fixed
        # penalty at every FOV and cannot move the minimum - the good GCPs decide it.
        # Capping (rather than simply ignoring outliers) still charges threshold^2 for
        # every GCP a FOV fails to fit, so a wrong FOV cannot 'win' by rejecting lots of
        # points and fitting the few that are left. The square root keeps it in pixels.
        capped = np.minimum(errors ** 2, ransac_threshold ** 2)                 # each GCP's squared error, capped at threshold^2
        score  = float(np.sqrt(capped.mean()))                                  # average, then back to pixels
        n_inl  = int((errors < ransac_threshold).sum())                         # how many GCPs this FOV fits within the threshold
        return score, n_inl, errors
    except Exception:                                 # otherwise catch any other error from OpenCV
        return 1e6, 0, None                           # and treat that trial as failed too


def estimate_fov_from_gcps(gcp_pixel, gcp_world, w_img, h_img,
                           fov_search_min=10.0, fov_search_max=120.0,
                           fov_coarse_step=1.0, fov_fine_range=3.0, fov_fine_step=0.1,
                           fov_ransac_threshold=20.0):
    """
    Option 3: search for the vertical FOV that best fits the GCPs.

    Returns
    -------
    fov         : best vertical FOV in degrees
    fov_est_err : robust reprojection error at that FOV, in pixels
    search      : dict of the search curves, for plot_fov_search
    """
    # Due to a quirk with the solvePnP code, it fails on large numbers such as
    # UTM coordinates (which are ~1e6). Before we do any calculations, we adjust the coords to help.
    # The mean of GCP world coordinates are subtracted from GCP to
    # artifically transform the numbers to help SolvePnP to work. This is rigid transformation
    # that does not affect the geometry. The coordinates are converted back by adding the
    # mean at the end. Cell 5 has to do the same thing.
    world_centre_est = gcp_world.mean(axis=0)               # axis=0 averages down the rows, giving one mean per column (E, N, Z)
    GCP_WORLD_C_est  = gcp_world - world_centre_est         # numpy subtracts the 3 means from every row at once ("broadcasting")

    # Coarse pass: sweep the whole range at 1 deg to find the 'basin' that the solution vertex sits in.
    print(f'Coarse search ({fov_search_min}-{fov_search_max} deg, '        # announce the coarse pass
          f'step {fov_coarse_step} deg)...')

    #build an array of test values
    fov_coarse = np.arange(fov_search_min,                                      # arange builds evenly spaced values: start,
                           fov_search_max + fov_coarse_step, fov_coarse_step)   # stop (exclusive, hence the +step), and step

    # Calculate robust reprojection error at all trial values in the fov_coarse array (call the function once per candidate FOV)
    # Each call returns (score, inlier count, per-GCP errors); keep the first two.
    # np.array() rather than a plain list, so the mask below can test every value at once.
    res_coarse = [reproj_error_at_fov(f, GCP_WORLD_C_est, gcp_pixel, w_img, h_img, fov_ransac_threshold)
                  for f in fov_coarse]
    err_coarse = np.array([r[0] for r in res_coarse])     # robust error per candidate
    inl_coarse = np.array([r[1] for r in res_coarse])     # GCPs agreeing per candidate
    valid_c    = err_coarse < 1e5              # mask out failed solves

    best_ci    = int(np.argmin(err_coarse))    # argmin gives the POSITION of the smallest value
    best_c_fov = float(fov_coarse[best_ci])    # the FOV at that position
    best_c_err = float(err_coarse[best_ci])    # the error at that position
    print(f'  Coarse best: {best_c_fov:.1f} deg  (error = {best_c_err:.2f} px, '
          f'{inl_coarse[best_ci]}/{len(gcp_pixel)} GCPs agree)')   # report it

    if best_c_err > 50:                        # even the best candidate is poor (more than 50 px average reprojection error)
        print('  WARNING: high error even at the best FOV — the GCPs probably do')  # the search still found nothing that fits well
        print('  not constrain the geometry. Check the GCPs before trusting this.')  # so the result should not be trusted

    # Fine pass: resolve the minimum to 0.1 deg within the coarse basin.
    fine_min = max(fov_search_min, best_c_fov - fov_fine_range)   # max() keeps the window inside the search range
    fine_max = min(fov_search_max, best_c_fov + fov_fine_range)   # min() does the same at the top end
    print(f'Fine search ({fine_min:.1f}-{fine_max:.1f} deg, step {fov_fine_step} deg)...')  # announce the fine pass and its window
    fov_fine = np.arange(fine_min, fine_max + fov_fine_step, fov_fine_step)   # candidates at 0.1 deg spacing
    res_fine = [reproj_error_at_fov(f, GCP_WORLD_C_est, gcp_pixel, w_img, h_img, fov_ransac_threshold)
                for f in fov_fine]                                            # error for each of them
    err_fine = np.array([r[0] for r in res_fine])
    inl_fine = np.array([r[1] for r in res_fine])
    valid_f  = err_fine < 1e5                                                 # mask of the ones that solved

    best_fi    = int(np.argmin(err_fine))      # position of the best fine candidate
    best_f_fov = float(fov_fine[best_fi])      # its FOV
    best_f_err = float(err_fine[best_fi])      # its error
    print(f'  Fine best  : {best_f_fov:.2f} deg  (error = {best_f_err:.2f} px, '
          f'{inl_fine[best_fi]}/{len(gcp_pixel)} GCPs agree)')  # report the best fine candidate

    # If the fine pass came out much worse, solvePnP became unstable there —
    # keep the coarse answer rather than a bad refinement.
    if best_f_err > best_c_err * 1.5:                                           # more than 50% worse than the coarse best
        print(f'  WARNING: fine search error ({best_f_err:.2f} px) far exceeds the')        # the fine pass went badly wrong
        print(f'  coarse best ({best_c_err:.2f} px) — falling back to the coarse value.')   # so the coarse answer is kept instead
        fov, fov_est_err = best_c_fov, best_c_err                          # assign two names at once from a pair
    else:                                                                       # the normal case: the fine pass improved on the coarse one
        fov, fov_est_err = best_f_fov, best_f_err                          # otherwise keep the fine result

    # Initial error assessment from the shape of the curve.
    # How well constrained is this? Take every FOV whose error is within 1 px of
    # the minimum: if that set is narrow the minimum is sharp, if it spans tens
    # of degrees the GCPs barely care what the FOV is.
    # (The leave-one-out test that follows gives the FOV uncertainty used downstream.)
    all_fovs  = np.concatenate([fov_coarse, fov_fine])      # join the two candidate arrays end to end
    all_errs  = np.concatenate([err_coarse, err_fine])      # and their two error arrays, keeping them aligned
    all_valid = all_errs < 1e5                              # a True/False array marking the successful solves
    tol_px    = 1.0                                         # "close to the minimum" means within this many pixels
    within_tol = all_fovs[all_valid & (all_errs < fov_est_err + tol_px)]                        # filter array to keep every FOV whose error is within tol_px of the best error found (the floor of the curve)
    fov_unc    = float(within_tol.max() - within_tol.min()) if len(within_tol) > 1 else 0.0     # Width of that set = how uncertain the FOV is. 0 if only one candidate qualified.

    print(f'\nCurve width     : +/-{fov_unc / 2:.2f} deg (within {tol_px} px of the minimum)')  # plus/minus half of the reprojection error curve width is first estimate of FOV uncertainty. Report here.

    search = {'fov_coarse': fov_coarse, 'err_coarse': err_coarse, 'inl_coarse': inl_coarse, 'valid_c': valid_c,
              'fov_fine': fov_fine, 'err_fine': err_fine, 'inl_fine': inl_fine, 'valid_f': valid_f,
              'within_tol': within_tol, 'fov_unc': fov_unc, 'tol_px': tol_px,
              'fov_coarse_step': fov_coarse_step, 'fov_fine_step': fov_fine_step,
              'fov_ransac_threshold': fov_ransac_threshold, 'n_gcps': len(gcp_pixel),
              'fov': fov, 'fov_est_err': fov_est_err}
    return fov, fov_est_err, search


# Function to report how well the FOV is constrained.
def print_fov_confidence(fov_sigma_deg, source):
    """
    Confidence judged on the full width of the uncertainty range (2 x sigma).

    source says where sigma came from ('leave-one-out' or 'curve width'). The
    leave-one-out sigma is preferred: the curve width is only a rough guide, and
    with the capped score it overstates the uncertainty, because the fixed penalty
    for any outlier compresses the curve.
    """
    fov_range = 2.0 * fov_sigma_deg                                             # full width of the +/- range, deg
    print(f'Confidence (from {source}, +/-{fov_sigma_deg:.2f} deg):')
    if fov_range < 5:     #degrees
        print('Confidence: HIGH — sharp minimum, FOV well constrained by the GCPs')
    elif fov_range < 15:  #degrees
        print('Confidence: MODERATE — reasonably constrained')
    else:
        print('Confidence: LOW — broad minimum; the GCPs do not constrain FOV.')
        print('  Find the focal length or sensor size if at all possible, or add')
        print('  GCPs spanning a wider elevation range.')


# Function for a quick FOV search in a window around a known estimate. Used by the
# leave-one-out test, which has to repeat the search once per GCP.
def fov_search_window(gcp_pixel, gcp_world, w_img, h_img, centre_fov, window=10.0,
                      coarse_step=1.0, fine_step=0.1, ransac_threshold=20.0):
    """
    Coarse + fine FOV search within centre_fov +/- window, without printing.

    Same scoring as the main search. The window keeps each repeat fast: dropping one
    GCP moves the FOV by a few degrees at most unless that GCP was holding the whole
    solution up, which the at_edge flag reports.

    Returns
    -------
    fov     : best FOV in degrees
    at_edge : True if the coarse best sat on the window edge (the true minimum may lie outside it)
    """
    gcp_world_c = gcp_world - gcp_world.mean(axis=0)                            # centre the coordinates, as in the main search

    # Coarse pass across the window
    fov_c = np.arange(centre_fov - window, centre_fov + window + coarse_step, coarse_step)
    fov_c = fov_c[(fov_c > 1) & (fov_c < 179)]                                  # drop impossible values
    err_c = np.array([reproj_error_at_fov(f, gcp_world_c, gcp_pixel, w_img, h_img, ransac_threshold)[0]
                      for f in fov_c])
    i_c     = int(np.argmin(err_c))
    best_c  = float(fov_c[i_c])
    at_edge = i_c in (0, len(fov_c) - 1)                                        # minimum on the boundary = may not be the real minimum

    # Fine pass: one coarse step either side of the coarse best
    fov_f = np.arange(best_c - coarse_step, best_c + coarse_step + fine_step / 2, fine_step)
    err_f = np.array([reproj_error_at_fov(f, gcp_world_c, gcp_pixel, w_img, h_img, ransac_threshold)[0]
                      for f in fov_f])
    if err_f.min() > err_c[i_c] * 1.5:                                          # same fallback as the main search
        return best_c, at_edge
    return float(fov_f[int(np.argmin(err_f))]), at_edge


def leave_one_out_fov(gcp_pixel, gcp_world, gcp_labels, fov, w_img, h_img,
                      ransac_threshold=20.0, window=10.0, coarse_step=1.0, fine_step=0.1):
    """
    Leave-one-out ('jackknife') test of the FOV estimate.

    The search is repeated once per GCP with that GCP removed. Two things come out:

    1. How much each GCP matters. A GCP whose removal shifts the FOV a long way is
       holding the answer up on its own - check it. (A GCP already flagged as an
       outlier usually shifts it very little, because the robust scoring was already
       ignoring it; its large residual is the warning sign instead.)
    2. An uncertainty for the FOV. The spread of the n leave-one-out answers gives
       the jackknife standard error:

           sigma = sqrt( (n - 1) / n * sum( (fov_i - mean(fov_i))^2 ) )

       This is the FOV uncertainty carried into the line uncertainty in Cell 13.

    Returns
    -------
    fov_sigma_deg : jackknife standard error of the FOV, in degrees
    loo           : dict of GCP label -> {'fov', 'shift', 'residual', 'outlier', 'at_edge'}
    """
    n = len(gcp_labels)

    # A pose needs 6 GCPs, so each repeat (with one removed) needs 7 to start with.
    if n < 7:
        print(f'\nLeave-one-out skipped: needs at least 7 GCPs (have {n}).')
        return None, {}

    print(f'\nLeave-one-out: repeating the search {n} times, each without one GCP '
          f'(window +/-{window:.0f} deg)...')

    # Residual of every GCP at the chosen FOV, from the same robust solve.
    _, _, resid = reproj_error_at_fov(fov, gcp_world - gcp_world.mean(axis=0), gcp_pixel,
                                      w_img, h_img, ransac_threshold)

    # Repeat the search with each GCP removed in turn
    loo_fov, loo_edge = [], []
    for i in range(n):
        keep = [j for j in range(n) if j != i]                                  # every GCP except number i
        f_i, edge = fov_search_window(gcp_pixel[keep], gcp_world[keep], w_img, h_img, fov,
                                      window, coarse_step, fine_step, ransac_threshold)
        loo_fov.append(f_i)
        loo_edge.append(edge)
    loo_fov = np.array(loo_fov)
    shifts  = loo_fov - fov                                                     # how far the answer moved without each GCP

    # Which GCPs the robust fit is ignoring (residual over the threshold at the best FOV)
    outlier = (resid >= ransac_threshold) if resid is not None else np.zeros(n, dtype=bool)

    # Jackknife standard error: the spread of the leave-one-out answers, scaled by
    # (n-1)/n because each answer shares all but one GCP with the others.
    # Only the GCPs the fit uses count. An outlier is already ignored by the robust
    # scoring, so removing it barely moves the FOV - its near-zero shift would only
    # dilute the spread and understate the uncertainty.
    use   = ~outlier
    n_use = int(use.sum())
    if n_use < 2:                                                               # not enough to measure a spread
        use, n_use = np.ones(n, dtype=bool), n
    fov_sigma_deg = float(np.sqrt((n_use - 1) / n_use *
                                  np.sum((loo_fov[use] - loo_fov[use].mean()) ** 2)))

    # Flag a GCP as influential if removing it shifts the FOV by more than 3x the
    # typical (median) shift, and by at least 1 degree.
    infl_limit = max(1.0, 3.0 * float(np.median(np.abs(shifts))))

    # Report as a table
    print(f'  {"GCP":<12} {"residual (px)":>13} {"FOV without":>12} {"shift":>8}')
    print(f'  {"-"*50}')
    loo = {}
    for i, label in enumerate(gcp_labels):
        is_out = bool(outlier[i])
        flag = ''
        if is_out:
            flag += '  <- outlier: ignored by the robust fit, check it'
        if abs(shifts[i]) > infl_limit:
            flag += '  <- influential: FOV depends on this GCP'
        if loo_edge[i]:
            flag += '  <- hit window edge'
        r_txt = f'{resid[i]:13.2f}' if resid is not None else f'{"-":>13}'
        print(f'  {label:<12} {r_txt} {loo_fov[i]:12.2f} {shifts[i]:+8.2f}{flag}')
        loo[label] = {'fov': float(loo_fov[i]), 'shift': float(shifts[i]),
                      'residual': float(resid[i]) if resid is not None else None,
                      'outlier': bool(is_out), 'at_edge': bool(loo_edge[i])}

    print(f'\nFOV uncertainty (leave-one-out, {n_use} of {n} GCPs): +/-{fov_sigma_deg:.2f} deg')
    if any(loo_edge):
        print('  WARNING: some repeats hit the edge of the search window. Widen')
        print('  FOV_LOO_WINDOW - that GCP may be holding the whole solution up.')
    print('  To drop a GCP, add it to EXCLUDE_GCPS in Cell 3 and re-run from there.')
    return fov_sigma_deg, loo


def plot_fov_search(search):
    """Plot robust reprojection error (and GCPs agreeing) against FOV from estimate_fov_from_gcps."""
    fov_coarse, err_coarse, valid_c = search['fov_coarse'], search['err_coarse'], search['valid_c']
    fov_fine, err_fine, valid_f     = search['fov_fine'], search['err_fine'], search['valid_f']
    tol_px                          = search['tol_px']
    fov, fov_est_err                = search['fov'], search['fov_est_err']
    FOV_COARSE_STEP, FOV_FINE_STEP  = search['fov_coarse_step'], search['fov_fine_step']
    inl_coarse, n_gcps              = search['inl_coarse'], search['n_gcps']
    fov_sigma    = search.get('fov_sigma', search['fov_unc'] / 2)               # FOV uncertainty from resolve_fov
    sigma_source = search.get('sigma_source', 'curve width')

    # Plot the error curve to see the shape (ideally narrow parabola, wider parabola is less constrained by GCPs)
    fig, ax = plt.subplots(figsize=(10, 5))                                  # one figure, 10 x 5 inches
    # np.where(mask, a, b) picks from a where the mask is True and b elsewhere;
    # np.nan means "not a number", which matplotlib leaves as a gap in the line.
    ax.plot(fov_coarse, np.where(valid_c, err_coarse, np.nan),
            color='steelblue', linewidth=1.5, label=f'Coarse ({FOV_COARSE_STEP} deg steps)')   # label feeds the legend
    ax.plot(fov_fine, np.where(valid_f, err_fine, np.nan),                   # the fine curve on the same axes
            color='darkblue', linewidth=1.2, label=f'Fine ({FOV_FINE_STEP} deg steps)')
    ax.axvline(fov, color='red', linewidth=2, linestyle='--',           # a vertical line at the chosen FOV
               label=f'Best FOV = {fov:.2f} deg (error = {fov_est_err:.2f} px)')
    ax.axhspan(fov_est_err, fov_est_err + tol_px, alpha=0.15, color='orange',   # a horizontal band 1 px tall; alpha is opacity
               label=f'Within {tol_px} px of the minimum')
    if fov_sigma > 0:                                                        # the FOV uncertainty carried downstream
        ax.axvspan(fov - fov_sigma, fov + fov_sigma, alpha=0.10, color='red',   # a vertical band spanning the uncertain range
                   label=f'FOV uncertainty ({sigma_source}): +/-{fov_sigma:.2f} deg')
    ax.set_xlabel('Vertical FOV (degrees)')                                  # x axis label
    ax.set_ylabel('Robust GCP reprojection error (px)')                      # y axis label
    ax.set_title('Reprojection error vs FOV — sharp narrow minimum = well constrained')   # title
    valid_errs = err_coarse[valid_c]                                         # the errors that came from successful solves
    # Cap the y axis so a few huge failures do not squash the interesting part
    # of the curve: 1.5x the 90th percentile, but never above 100.
    ax.set_ylim(bottom=0,
                top=min(float(np.percentile(valid_errs, 90)) * 1.5
                        if len(valid_errs) else 50, 100))
    ax.grid(alpha=0.3)                                                       # faint gridlines

    # Second y axis: how many GCPs each FOV fits within the threshold. It should
    # peak around the best FOV; a wrong FOV loses GCPs.
    ax2 = ax.twinx()                                                         # shares the x axis, own y axis on the right
    ax2.step(fov_coarse, np.where(valid_c, inl_coarse, np.nan), where='mid',
             color='grey', linewidth=1, alpha=0.7, label='GCPs agreeing (right axis)')
    ax2.set_ylabel(f'GCPs within {search["fov_ransac_threshold"]:g} px')
    ax2.set_ylim(0, n_gcps + 0.5)

    # One legend for both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9)                                  # draw the legend from the labels above
    plt.tight_layout()                                                       # tidy spacing
    plt.show()                                                               # display


def resolve_fov(gcp_pixel, gcp_world, w_img, h_img, aspect,
                focal_length_mm=None, sensor_width_mm=None, sensor_height_mm=None,
                vertical_fov_deg=None, fov_uncertainty_deg=0.0, estimate_from_gcps=True,
                fov_search_min=10.0, fov_search_max=120.0,
                fov_coarse_step=1.0, fov_fine_range=3.0, fov_fine_step=0.1,
                fov_ransac_threshold=20.0, fov_leave_one_out=True, fov_loo_window=10.0,
                gcp_labels=None):
    """
    Resolve the vertical FOV from Option 1 (focal length + sensor), Option 2
    (FOV given directly) or Option 3 (estimate from the GCPs), in that order.

    Returns
    -------
    fov           : vertical FOV in degrees
    search        : dict of search curves for plot_fov_search (Option 3), else None
    fov_sigma_deg : FOV uncertainty in degrees (Options 1/2: fov_uncertainty_deg;
                    Option 3: leave-one-out standard error), used in Cell 13
    """
    search = None
    fov_sigma_deg = 0.0

    # ── Option 1: derive FOV from the optics ──────────────────────────────────────
    if focal_length_mm is not None and sensor_width_mm is not None:                     # If you've entered values for focal length and sensor width, option 1 will run
        sensor_h = (sensor_height_mm if sensor_height_mm is not None                    # Vertical FOV needs sensor height. If only the width is known, assume the
                    else sensor_width_mm / aspect)                                      # sensor has the same aspect ratio as the image (true unless the image was cropped
        if sensor_height_mm is None:
            print(f'Sensor height derived from image aspect ratio: {sensor_h:.3f} mm')  # Print the source of sensor height to check

        fov = focal_and_sensor_to_fov(focal_length_mm, sensor_h)                   # call the vertical FOV function
        fov_sigma_deg = float(fov_uncertainty_deg or 0.0)                          # known optics: uncertainty as entered
        print('Option 1 — focal length + sensor size:')                                 # report which option was used
        print(f'  Focal length : {focal_length_mm} mm')                                 # echo the focal length
        print(f'  Sensor size  : {sensor_width_mm} x {sensor_h:.3f} mm')                # echo the sensor dimensions

    # ── Option 2: FOV given directly ──────────────────────────────────────────────
    elif vertical_fov_deg is not None:                                                  # If you've entered values for FOV directly, option 2 will run
        fov = float(vertical_fov_deg)                                              # float() guards against the value being typed as an integer
        fov_sigma_deg = float(fov_uncertainty_deg or 0.0)                          # known FOV: uncertainty as entered
        print(f'Option 2 — FOV supplied directly: {fov:.2f} deg')                  # echo the FOV

    # ── Option 3: search for the FOV that best fits the GCPs ──────────────────────
    elif estimate_from_gcps:                                                            # If all are none, then begin estimating the FOV.
        print('Option 3 — estimating FOV from GCPs')                                    # reporting that this is what is happening, and what bounds were entered
        print(f'  Search : {fov_search_min}-{fov_search_max} deg  '
              f'(coarse {fov_coarse_step} deg, fine {fov_fine_step} deg)')
        print(f'  GCPs   : {len(gcp_pixel)}   Robust threshold: {fov_ransac_threshold} px\n')
        fov, _, search = estimate_fov_from_gcps(
            gcp_pixel, gcp_world, w_img, h_img,
            fov_search_min, fov_search_max, fov_coarse_step, fov_fine_range, fov_fine_step,
            fov_ransac_threshold)

        # Leave-one-out test: which GCPs the answer depends on, and the FOV uncertainty.
        loo_sigma = None
        if fov_leave_one_out:
            labels = gcp_labels if gcp_labels is not None else [f'GCP {i+1}' for i in range(len(gcp_pixel))]
            loo_sigma, search["loo"] = leave_one_out_fov(
                gcp_pixel, gcp_world, labels, fov, w_img, h_img,
                fov_ransac_threshold, fov_loo_window, fov_coarse_step, fov_fine_step)
        if loo_sigma is not None:
            fov_sigma_deg, sigma_source = loo_sigma, 'leave-one-out'
        else:
            # Without leave-one-out, fall back on half the width of the curve's floor.
            fov_sigma_deg, sigma_source = search['fov_unc'] / 2, 'curve width'
            print(f'\nFOV uncertainty taken from the curve width: +/-{fov_sigma_deg:.2f} deg')
        search['fov_sigma'], search['sigma_source'] = fov_sigma_deg, sigma_source   # for the plot
        print_fov_confidence(fov_sigma_deg, sigma_source)

    # ── Nothing set - fall back to a generic value  ───────────────
    else:
        fov = 55.0
        print('WARNING: no camera specs given and ESTIMATE_FOV_FROM_GCPS = False.')
        print('  Defaulting to 55 deg, which is almost certainly wrong.')
        print('  Set FOCAL_LENGTH_MM + SENSOR_WIDTH_MM, or VERTICAL_FOV_DEG,')
        print('  or ESTIMATE_FOV_FROM_GCPS = True.')

    return fov, search, fov_sigma_deg


def report_fov(fov, h_img, fov_sigma_deg=0.0):
    """Print the chosen FOV and focal length with a plausibility check. Returns focal_px."""
    # ── Report and sanity check ───────────────────────────────────────────────────
    focal_px = fov_to_focal_px(fov, h_img)        # convert the chosen FOV into focal length (pixels)
    print(f'\n  Vertical FOV   : {fov:.2f} +/- {fov_sigma_deg:.2f} deg')   # print estimated FOV and its uncertainty
    print(f'  Focal (pixels) : {focal_px:.1f} px')     # print estimated focal length (pixels)

    #print some warnings if the FOV seems too wide or too narrow.
    if fov < 5 or fov > 170:
        print(f'\nWARNING: {fov:.1f} deg is not a plausible FOV — check the inputs.')
    elif fov < 15:
        print(f'\nNOTE: very narrow FOV ({fov:.1f} deg) — telephoto lens?')
    elif fov > 120:
        print(f'\nNOTE: very wide FOV ({fov:.1f} deg) — fisheye lens? This pipeline')
        print('  assumes a rectilinear lens and does not account for fisheye distortion.')
    else:
        print('\nFOV looks plausible. Continue to Cell 5.')
    return focal_px


# ══════════════════════════════════════════════════════════════════════════════
# Cell 5 — Solve camera pose
# ══════════════════════════════════════════════════════════════════════════════

def build_camera_matrix(fov, w_img, h_img):
    """
    Build the camera intrinsic matrix K from the vertical FOV.

    Returns K, focal_px, dist_coeffs.
    """
    # ── Build the camera intrinsic matrix K ───────────────────────────────────────
    # K maps a 3D point in camera coordinates to a 2D pixel. It holds the focal
    # length in pixels and the 'principal point' (assumed to be the image centre).
    # The -0.5 puts the principal point at the centre of the centre pixel, matching
    # OpenCV's convention that pixel (0,0) is centred on the coordinate (0,0).
    focal_px = fov_to_focal_px(fov, h_img)         # calculate focal length in pixels (function in Cell 4)
    cx_img   = w_img / 2.0 - 0.5                        # principal point x: horizontal image centre
    cy_img   = h_img / 2.0 - 0.5                        # principal point y: vertical image centre

    K = np.array([[focal_px, 0,        cx_img],      # row 1: focal length and the principal point's x
                     [0,        focal_px, cy_img],      # row 2: focal length and the principal point's y
                     [0,        0,        1.0   ]],     # row 3: fixed, copies z through for the perspective divide (col = f*x/z + cx)
                     dtype=np.float64)                  # dtype sets 64-bit precision

    # Lens distortion is assumed to be zero. If the lens is strongly wide-angle and
    # the reprojection error stays high, try putting real coefficients here. Manufacturer values,
    # calibration using a chequerboard, or tests from the lensfun.github.io database will work if lens is known.
    # Order is (k1, k2, p1, p2).
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0])

    return K, focal_px, dist_coeffs


def solve_pose(gcp_pixel, gcp_world, gcp_labels, K, dist_coeffs, ransac_threshold,
               fov, focal_px):
    """
    Solve the camera position and orientation from the GCPs with solvePnPRansac.

    Returns
    -------
    R         : (3, 3) rotation matrix, world -> camera
    cam_world : (3,)   camera position [E, N, Z] in the DEM's CRS
    uv          : (N, 2) model-projected GCP pixel positions
    err         : (N,)   per-GCP reprojection error in pixels (every GCP)
    inlier_mask : (N,)   True for the GCPs RANSAC kept - the ones the pose was solved from
    """
    # ── Centre the world coordinates ──────────────────────────────────────────────
    # solvePnP works in double precision but conditions badly on coordinates of
    # ~1e6 (typical UTM eastings). Subtracting the GCP centre brings everything to
    # a few hundred metres without changing the geometry. The offset is added back
    # after the solve. Downstream cells always use full world coordinates.
    world_centre = gcp_world.mean(axis=0)           # mean easting, northing and elevation of all GCPs
    GCP_WORLD_C  = gcp_world - world_centre         # every row shifted by that centroid

    print(f'World centre offset: E={world_centre[0]:.1f}  '         #report the offset that was applied
          f'N={world_centre[1]:.1f}  Z={world_centre[2]:.1f}')      #so that camera position can be traced if needed

    # ── Solve the pose ────────────────────────────────────────────────────────────
    # Four values come back: a success flag, the rotation, the translation, and the
    # list of GCPs the solver decided to trust. Uses solvePnPRansac which is a version
    # that includes RANSAC outlier detection.
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        GCP_WORLD_C.reshape(-1, 1, 3).astype(np.float64),   # 3D points (centred)
        gcp_pixel.reshape(-1, 1, 2).astype(np.float64),     # matching 2D pixels
        K, dist_coeffs,                        # the intrinsics and distortion built above
        iterationsCount=10000,                # random subsets to try
        reprojectionError=ransac_threshold,   # inlier tolerance, px
        confidence=0.999,                     # stop early once this confident
        flags=cv2.SOLVEPNP_ITERATIVE)         # which solvepnp algorithm OpenCV should use

    if not success:
        # RANSAC needs a workable consensus set: at least 5 GCPs that agree within
        # ransac_threshold px. If it cannot find one, stop rather than falling back to a
        # plain solve on every GCP - that would quietly bring back the outliers RANSAC is
        # there to reject, and the pose would shift with every GCP added or removed.
        # For diagnosis only, show how far each GCP sits from a plain fit to all of them.
        print(f'RANSAC failed: fewer than 5 GCPs agree within {ransac_threshold} px.')
        try:
            ok_d, rvec_d, tvec_d = cv2.solvePnP(                        # plain fit, for the table below only
                GCP_WORLD_C.reshape(-1, 1, 3).astype(np.float64),
                gcp_pixel.reshape(-1, 1, 2).astype(np.float64),
                K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok_d:
                uv_d, _ = cv2.projectPoints(GCP_WORLD_C.reshape(-1, 1, 3), rvec_d, tvec_d, K, dist_coeffs)
                err_d = np.linalg.norm(gcp_pixel - uv_d.reshape(-1, 2), axis=1)
                print('  For reference, residuals from a plain fit to ALL GCPs (not used):')
                for label, e in zip(gcp_labels, err_d):
                    print(f'    {label:<12} {e:>8.1f} px')
        except Exception:
            pass                                                        # the table is optional
        raise RuntimeError(f'No reliable camera pose: fewer than 5 GCPs agree within '
                           f'RANSAC_THRESHOLD = {ransac_threshold} px. Either raise RANSAC_THRESHOLD '
                           f'(and FOV_RANSAC_THRESHOLD in Cell 4) to suit your GCP accuracy, or '
                           f'check and exclude bad GCPs in Cell 3.')

    # print report
    print(f'\nsolvePnP: {len(inliers)}/{len(gcp_pixel)} GCPs used as inliers')      # how many points the solver trusted
    if len(inliers) < len(gcp_pixel):                                               # if fewer kept than supplied, something was rejected
        outlier_labels = [gcp_labels[i] for i in range(len(gcp_pixel))              # inliers holds the positions RANSAC kept, as an (M,1) column. This inverts it - any position absent from that list was an outlier.
                          if i not in inliers.flatten()]                            # .flatten() is for readability
        print(f'  Rejected as outliers: {outlier_labels}')
        print('  Check these points ')
    if len(inliers) < 6:                                                            # the pose is only as good as the few GCPs left
        print(f'  WARNING: the pose rests on only {len(inliers)} GCPs. Adding or removing a single')
        print('  GCP can move it a long way. Add GCPs, or raise RANSAC_THRESHOLD if good GCPs')
        print('  are being rejected.')


    # ── Recover the camera position in world coordinates ──────────────────────────
    # solvePnP returns the pose as world -> camera:  x_cam = R @ x_world + t
    # The camera centre is therefore C = -R^T @ t, in centred coordinates. Adding
    # world_centre puts it back in the DEM's CRS.


    R         = cv2.Rodrigues(rvec)[0]                      # take the solved camera rotation matrix and convert to a 3x3 table using cv2.Rodrigues method #This is the view of camera looking out to world.
    cam_world = (-R.T @ tvec).ravel() + world_centre        # flip the rotation to get world looking back at camera. Uses .T to transpose/flip the view rotation matrix, - to reverse the direction from the world origin 'tvec' matrix. Also addin back the GCP centre to convert to real coords
    cam_x, cam_y, cam_z = cam_world                         # unpack the three values into separate names


    # ── Final reprojection error ──────────────────────────────────────────────────
    # uv ('u' = across, 'v' = down in standard notation) holds pixel positions where the model thinks each GCP should appear.
    # against where it was actually picked.
    reproj_final, _ = cv2.projectPoints(GCP_WORLD_C.reshape(-1, 1, 3),                  # project the points from the solved camera out into the world
                                        rvec, tvec, K, dist_coeffs)
    uv = reproj_final.reshape(-1, 2)                                                    # flatten from (N,1,2) to (N,2). One [col, row] per GCP
    err = np.linalg.norm(gcp_pixel - uv, axis=1)                                        # per-GCP error (px). Axis=1 turns across + down into striaght line distance error using trigonometry

    # ── Split the error into the GCPs the pose was fitted to, and the rest ────────
    # The pose was solved from RANSAC's inliers only, so the camera model's error is
    # judged on those: the inlier mean is what goes into the uncertainty downstream.
    # err keeps every GCP, so outliers stay visible in the table below and in Cell 6.
    inlier_mask = np.zeros(len(gcp_pixel), dtype=bool)                                 # one True/False per GCP
    inlier_mask[np.asarray(inliers).flatten()] = True                                  # True for the GCPs RANSAC kept
    err_inliers = err[inlier_mask]                                                     # errors of the GCPs the pose was fitted to

    ## Report results
    print(f'\nCamera position : E={cam_x:.1f}  N={cam_y:.1f}  Z={cam_z:.1f}')           # camera position X, Y, Z
    print(f'Vertical FOV    : {fov:.3f} deg   (focal {focal_px:.1f} px)')               # FOV, focal length
    print(f'Reprojection    : mean {err_inliers.mean():.2f} px   max {err_inliers.max():.2f} px   '    # reprojection error mean and max
          f'({inlier_mask.sum()} inlier GCPs - used downstream)')
    print(f'                  mean {err.mean():.2f} px   max {err.max():.2f} px   '     # and over every GCP, for transparency
          f'(all {len(err)} GCPs - for reference)')

    print(f'\nPer-GCP reprojection error:')                                             #print formatted table of individual GCPs and errors
    print(f'  {"GCP":<12} {"error (px)":>10}')
    print(f'  {"-"*24}')
    for label, e, keep in zip(gcp_labels, err, inlier_mask):
        print(f'  {label:<12} {e:>10.2f}' + ('' if keep else '  <- RANSAC outlier (not used in pose)'))

    #Print warning if mean reprojection error is high (more than 20 px, arbitrary threshold).
    if err_inliers.mean() > 20:
        print(f'\nHigh reprojection error ({err_inliers.mean():.1f} px). Check the Cell 6 plot,')
        print('  can GCPs be improved or excluded?')
    else:
        print('Check the Cell 6 plot.')

    return R, cam_world, uv, err, inlier_mask


# ══════════════════════════════════════════════════════════════════════════════
# Cell 6 — Validate GCP reprojection
# ══════════════════════════════════════════════════════════════════════════════

def print_reprojection_table(gcp_labels, gcp_pixel, uv, err, inlier_mask=None):
    """Print picked vs projected GCP positions and their errors (inlier_mask from Cell 5)."""
    if inlier_mask is None:
        inlier_mask = np.ones(len(err), dtype=bool)
    # uv (model-projected GCP positions) and err came from Cell 5.
    # Print table of picked vs projected GCPs.
    print(f'  {"GCP":<12} {"picked (col,row)":>20}  {"projected":>20}  {"error (px)":>10}')
    print(f'  {"-"*68}')
    for label, px, pp, e, keep in zip(gcp_labels, gcp_pixel, uv, err, inlier_mask):
        flag = '  <- high' if e > 10 else ('  <- check' if e > 5 else '')
        if not keep:
            flag += '  <- RANSAC outlier (not used in pose)'
        print(f'  {label:<12} ({px[0]:>7.1f},{px[1]:>7.1f})  '
              f'({pp[0]:>7.1f},{pp[1]:>7.1f})  {e:>10.2f}{flag}')
    # Inlier mean is the camera model's error (used downstream); all-GCP mean for reference
    print(f'\n  Inlier GCPs ({inlier_mask.sum()}): Mean: {err[inlier_mask].mean():.2f} px   '
          f'Max: {err[inlier_mask].max():.2f} px')
    print(f'  All GCPs ({len(err)}):    Mean: {err.mean():.2f} px   Max: {err.max():.2f} px')


def plot_reprojection(img_rgb, gcp_labels, gcp_pixel, uv, err, inlier_mask=None):
    """Draw picked vs projected GCP positions joined by an error line (inlier_mask from Cell 5)."""
    if inlier_mask is None:
        inlier_mask = np.ones(len(err), dtype=bool)
    # ── Plot picked vs projected, joined by an error line ─────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img_rgb)
    for label, px, pp, e, keep in zip(gcp_labels, gcp_pixel, uv, err, inlier_mask):
        # Only label the first point of each type so the legend has two entries.
        ax.plot(px[0], px[1], 'r+', markersize=14, markeredgewidth=2,
                label='Picked' if label == gcp_labels[0] else '')
        ax.plot(pp[0], pp[1], 'bo', markersize=10, fillstyle='none', markeredgewidth=1.5,
                label='Projected' if label == gcp_labels[0] else '')
        ax.plot([px[0], pp[0]], [px[1], pp[1]], 'y-', linewidth=1)
        ax.annotate(f'{label}\n{e:.1f}px' + ('' if keep else '\n(outlier)'), xy=(px[0], px[1]), xytext=(6, -16),
                    textcoords='offset points', fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
    ax.legend(loc='upper right')
    ax.set_title(f'GCP reprojection — red = picked, blue = projected  |  '
                 f'mean error = {err[inlier_mask].mean():.2f} px ({inlier_mask.sum()} inliers), '
                 f'{err.mean():.2f} px (all {len(err)})')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Cell 7 — Load DEM
# ══════════════════════════════════════════════════════════════════════════════

def load_dem(dem_path, output_res_m=None, cam_world=None):
    """
    Load the DEM and define the output raster grid.

    Parameters
    ----------
    dem_path     : DEM GeoTIFF, projected CRS in metres
    output_res_m : output pixel size in metres, or None to keep the DEM's own
    cam_world    : camera position from Cell 5 (optional), used to report
                   whether the camera sits inside the DEM extent

    Returns
    -------
    dem : dict holding dem_data, dem_transform, dem_crs, dem_nodata, dem_res_x,
          dem_bounds, out_h, out_w, out_transform, out_crs
    """
    # ──────────────────────────────────────────────────────────────────────────────
    # Load DEM and copy out its data which is used downstream
    # "with ... as ds" opens the raster, names it ds, and closes it at the end of the indented block
    print(f'Loading DEM: {dem_path}')
    with rasterio.open(dem_path) as ds:
        dem_data      = ds.read(1).astype(np.float64)   # elevation grid is band 1
        dem_transform = ds.transform                    # grid -> world affine conversion
        dem_crs       = ds.crs                          # coordinate reference system
        dem_nodata    = ds.nodata                       # void value, if declared
        dem_res_x     = abs(ds.transform.a)             # pixel size in metres
        dem_bounds    = ds.bounds                       # extent in world coordinates

    # Define the output raster grid, which starts as a copy of the dem
    out_h, out_w  = dem_data.shape
    out_transform = dem_transform
    out_crs       = dem_crs

    # report raster information for sanity check
    print(f'  Size   : {out_w} x {out_h} px')
    print(f'  Res    : {dem_res_x:.2f} m/pixel')
    print(f'  CRS    : {dem_crs}')
    print(f'  Bounds : E {dem_bounds.left:.1f}-{dem_bounds.right:.1f}  '
          f'N {dem_bounds.bottom:.1f}-{dem_bounds.top:.1f}')


    # ── Optional resample to a different output resolution ────────────────────────
    if output_res_m is not None and output_res_m != dem_res_x:    # if a different size was actually asked for
        scale = dem_res_x / output_res_m                          # >1 makes the grid finer, <1 coarser
        out_w = int(out_w * scale)                                # int() drops the fractional part
        out_h = int(out_h * scale)                                # same for the height
        # Same geographic extent, different number of cells.
        out_transform = from_bounds(dem_bounds.left, dem_bounds.bottom,       # build a new grid definition from
                                    dem_bounds.right, dem_bounds.top, out_w, out_h)   # the old extent and the new size
        dem_resampled = np.empty((out_h, out_w), dtype=np.float64)   # an empty array of the right shape to fill
        rasterio.warp.reproject(dem_data, dem_resampled,             # source array and destination array
                                src_transform=dem_transform, src_crs=dem_crs,     # where the source cells sit
                                dst_transform=out_transform, dst_crs=dem_crs,     # where the destination cells sit
                                resampling=rasterio.warp.Resampling.bilinear)     # blend the four nearest cells
        dem_data = dem_resampled                                  # from here on, "the DEM" means the resampled one
        # Point the DEM variables at the resampled grid too, so the mesh built in
        # Cell 8 uses the same geometry as the output raster.
        dem_transform = out_transform                             # keep the grid definition in step
        dem_res_x     = output_res_m                              # and the recorded cell size
        print(f'  Resampled -> {out_w} x {out_h} at {output_res_m} m')   # report the change


    # ── Checks and warnings ──────────────────────────────────────────────────────
    if dem_crs is None:
        print('\nWARNING: the DEM has no CRS. Outputs will not be georeferenced.')
    elif not dem_crs.is_projected:
        print('\nWARNING: the DEM CRS is geographic (degrees), not projected (metres).')
        print('  Reproject it to UTM or another metre-based CRS before continuing —')
        print('  every distance in this notebook assumes metres.')
    else:
        print('\nDEM loaded. Continue to Cell 8 to build the terrain mesh.')

    # Is the camera inside the DEM footprint? Not required (the camera can stand
    # outside the mapped area) but worth knowing.
    # Skipped if no camera position is passed in (Cell 5 not run yet).
    if cam_world is not None:
        cam_x, cam_y = cam_world[0], cam_world[1]
        inside = (dem_bounds.left <= cam_x <= dem_bounds.right and                          # camera easting within the DEM, and
                  dem_bounds.bottom <= cam_y <= dem_bounds.top)                             # camera northing within it too
        print(f'  Camera is {"inside" if inside else "OUTSIDE"} the DEM extent.')           # if statement made inside the f-string

    return {'dem_data': dem_data, 'dem_transform': dem_transform, 'dem_crs': dem_crs,
            'dem_nodata': dem_nodata, 'dem_res_x': dem_res_x, 'dem_bounds': dem_bounds,
            'out_h': out_h, 'out_w': out_w, 'out_transform': out_transform, 'out_crs': out_crs}


# ══════════════════════════════════════════════════════════════════════════════
# Cell 8 — Build terrain mesh from DEM
# ══════════════════════════════════════════════════════════════════════════════

def build_or_load_mesh(mesh_path, dem, mesh_stride=1, mesh_simplify=0.5):
    """
    Build a TIN mesh from the DEM and save it to mesh_path, or load it if the
    file already exists.

    Returns
    -------
    mesh  : Open3D TriangleMesh
    verts : (V, 3) vertex coordinates [E, N, Z]
    t     : (T, 3) triangle corner indices
    """
    # DEM values from Cell 7
    dem_data, dem_transform = dem['dem_data'], dem['dem_transform']
    dem_nodata, dem_res_x   = dem['dem_nodata'], dem['dem_res_x']
    out_h, out_w            = dem['out_h'], dem['out_w']

    # ──────────────────────────────────────────────────────────────────────────────
    # check if Open3d installed from the import cell
    if not OPEN3D_OK:
        raise ImportError('open3d is required for this cell: pip install open3d')

    # ── Reuse an existing mesh if one is already on disk ──────────────────────────
    if os.path.exists(mesh_path):
        print('Mesh already exists — loading it instead of rebuilding.')
        print(f'  {mesh_path}')
        print('  Delete or rename that file to rebuild with new settings.')

        # read the existing .obj file back into an Open3D mesh
        mesh  = o3d.io.read_triangle_mesh(mesh_path)

        #report size of mesh
        verts = np.asarray(mesh.vertices)                                       # read verticies as numpy array
        t = np.asarray(mesh.triangles)                                          # read triangles as numpy array
        n_v   = len(verts)                                                      # number of vertices
        n_t   = len(t)                                                          # number of triangles
        print(f'  {n_v:,} vertices   {n_t:,} triangles')                        # print summary (E in index 0, N is index 1, Z is index 2)
        print(f'  E: {verts[:,0].min():.1f} -> {verts[:,0].max():.1f}')
        print(f'  N: {verts[:,1].min():.1f} -> {verts[:,1].max():.1f}')
        print(f'  Z: {verts[:,2].min():.1f} -> {verts[:,2].max():.1f}')

    # ── Otherwise, create a new mesh from DEM ──────────────────────────
    else:
        print('Building terrain mesh from DEM...')                                                  # announce the beginning and report summary stats
        print(f'  DEM    : {out_w} x {out_h} px  res={dem_res_x:.2f} m')
        print(f'  Stride : {mesh_stride}  (effective spacing {dem_res_x*mesh_stride:.1f} m)')

        # ── Subsample the DEM grid ────────────────────────────────────────────────
        row_idx  = np.arange(0, dem_data.shape[0], mesh_stride)                 # row numbers to keep: 0, stride, 2*stride, ... (start, stop, step)
        col_idx  = np.arange(0, dem_data.shape[1], mesh_stride)                 # col numbers to keep: 0, stride, 2*stride, ... (start, stop, step)
        col_grid_m, row_grid_m = np.meshgrid(col_idx, row_idx)                  # meshgrid pairs every kept column with every kept row, giving two 2D arrays that together name each vertex's position in the grid.

        n_rows_s, n_cols_s = len(row_idx), len(col_idx)                         # count how many of each survived for summary stats

        # ── Turn grid indices into real-world vertex coordinates ──────────────────
        # The affine transform maps the corner of a cell, so +0.5 shifts to the cell
        # centre, where the elevation value actually applies. transform.e is
        # negative for a north-up raster, which is why northings decrease with row.
        # .a, .c, etc... are positions in the rasterio affine transform matrix
        xs = dem_transform.c + (col_grid_m + 0.5) * dem_transform.a             # .c is the origin easting, .a the cell width
        ys = dem_transform.f + (row_grid_m + 0.5) * dem_transform.e             # .f is the origin northing, .e the (negative) cell height
        zs = dem_data[row_grid_m, col_grid_m]                                   # pair the two index grids element by element. One elevation per vertex, same shape as xs and ys

        # Mark DEM voids as NaN so no triangle is built across them.
        if dem_nodata is not None:                                               # if the file declared a void value
            zs = np.where(zs == dem_nodata, np.nan, zs)                          # replace those cells with NaN, keep the rest

        # create vertices from x's, y's and z's.
        vertices_m = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)                 # .ravel() flattens a 2D array into one long 1D list. stack(..., axis=1) then glues the three lists into an (N, 3) table of [E, N, Z] rows.
        valid_v    = np.isfinite(zs).ravel()                                                # create list of true/false if vertice is real number (not NaN)
        print(f'  Grid   : {n_cols_s} x {n_rows_s} = {n_cols_s*n_rows_s:,} vertices  '      # print summary stats
              f'({valid_v.sum():,} valid)')

        # ── Build triangles: two per grid quad ────────────────────────────────────
        # For every 2x2 block of vertices, split the quad along one diagonal:
        #   v00 --- v01        triangle A = (v00, v10, v01)
        #    |  \    |         triangle B = (v10, v11, v01)
        #   v10 --- v11
        ii, jj = np.meshgrid(np.arange(n_rows_s - 1), np.arange(n_cols_s - 1),          # ii and jj hold the row and column of each quad's top-left corner
                             indexing='ij')                                             # indexing='ij' keeps them in row-major (matrix) order.
        ii, jj = ii.ravel(), jj.ravel()                                                 # flatten both to 1D

        # A vertex's position in the flat list is row * width + column.
        v00 = ii * n_cols_s + jj                                                        # top-left of each quad
        v10 = (ii + 1) * n_cols_s + jj                                                  # bottom-left
        v01 = ii * n_cols_s + (jj + 1)                                                  # top right
        v11 = (ii + 1) * n_cols_s + (jj + 1)                                            # bottom right
        tris_m = np.vstack([np.stack([v00, v10, v01], axis=1),                          # where stack makes each triangle a row of three vertex numbers, vstack puts the
                            np.stack([v10, v11, v01], axis=1)])                         # two sets of triangles one above the other so that open3d has the correct vertice format

        # Drop any triangle touching a void vertex.
        tris_m = tris_m[valid_v[tris_m].all(axis=1)]                                    # valid_v[tris_m] looks up all three corners at once; .all(axis=1) is True only where every corner of that row is valid.
        print(f'  Triangles before simplification: {len(tris_m):,}')                    # count before decimation

       # ── Assemble the Open3D mesh ──────────────────────────────────────────────
        mesh = o3d.geometry.TriangleMesh()                            # an empty mesh object
        mesh.vertices  = o3d.utility.Vector3dVector(vertices_m)       # hand it the vertex positions (3 doubles per row)
        mesh.triangles = o3d.utility.Vector3iVector(tris_m)           # and the triangle corner numbers (3 ints per row)
        mesh.remove_duplicated_vertices()                             # merge vertices that sit on top of each other
        mesh.remove_duplicated_triangles()                            # drop repeated triangles
        mesh.compute_vertex_normals()                                 # work out surface directions, needed for shading

        # ── Simplify ──────────────────────────────────────────────────────────────
        # Quadric decimation collapses edges whose removal changes the surface least,
        # so flat ground loses triangles and ridges keep them.
        if mesh_simplify < 1.0:                                             # skip if set to 1.0
            n_before = len(np.asarray(mesh.triangles))                      # count before
            target   = max(int(n_before * mesh_simplify), 100)              # keep this fraction of the triangles, converted to integer (whole number) (which rounds down), but never go below 100 (arbitrary minimum)
            print(f'  Simplifying {n_before:,} -> {target:,} triangles...') # report the reduction before it happens
            mesh = mesh.simplify_quadric_decimation(target)                 # returns a new, smaller mesh
            mesh.remove_duplicated_vertices()                               # tidy the result again
            mesh.remove_duplicated_triangles()                              # and again
            mesh.compute_vertex_normals()                                   # normals must be recomputed after decimation

        verts = np.asarray(mesh.vertices)                                   # final vertex table
        t = np.asarray(mesh.triangles)                                      # final triangle table
        n_v   = len(verts)                                                  # final vertex count
        n_t   = len(t)                             # final triangle count
        print(f'  Final  : {n_v:,} vertices  {n_t:,} triangles')            # report the result
        print(f'  E: {verts[:,0].min():.1f} -> {verts[:,0].max():.1f}')     # easting extent of the mesh
        print(f'  N: {verts[:,1].min():.1f} -> {verts[:,1].max():.1f}')     # northing extent
        print(f'  Z: {verts[:,2].min():.1f} -> {verts[:,2].max():.1f}')     # elevation range

        # ── Write the OBJ ─────────────────────────────────────────────────────────
        print(f'\nWriting -> {mesh_path}')                                       # announce what is happening
        o3d.io.write_triangle_mesh(mesh_path, mesh)                              # save it so later runs can reuse it
        print(f'  File size: {os.path.getsize(mesh_path) / 1e6:.1f} MB')         # report file size

        # ── Check approximate ground size of triangles ───────────────
        # verts[t[:, 1]] looks up the second corner of every triangle at once. The norm of
        # the difference between two corners is that edge's length in metres.
        all_edges = np.concatenate([
            np.linalg.norm(verts[t[:, 1]] - verts[t[:, 0]], axis=1),                 # first edge of every triangle
            np.linalg.norm(verts[t[:, 2]] - verts[t[:, 1]], axis=1),                 # second edge
            np.linalg.norm(verts[t[:, 0]] - verts[t[:, 2]], axis=1)])                # third edge
        print(f'  Edge lengths: min={all_edges.min():.1f} m  '                       # shortest edge
              f'median={np.median(all_edges):.1f} m  max={all_edges.max():.1f} m')   # typical and longest
        if np.median(all_edges) <= dem_res_x * 2:                                    # if typical triangle no wider than two DEM cells
            print('  Mesh density OK.')                                              # triangles are small enough to follow the terrain
        else:                                                                        # otherwise the mesh has been thinned a lot
            print('  Triangles are large relative to the DEM. Raise MESH_SIMPLIFY')  # and the mesh may be too coarse for fine detail
            print('  or lower MESH_STRIDE if fine terrain detail matters.')          # so the two settings that control it

        print('\nMesh built. Continue to Cell 9.')

    return mesh, verts, t


def plot_mesh(verts, t, mesh_stride, mesh_simplify):
    """Wireframe of the terrain mesh with a zoomed inset to check triangle density."""
    n_v, n_t = len(verts), len(t)

    # ── Visual check: wireframe with a zoomed inset ───────────────────────────────
    E_v, N_v, Z_v = verts[:, 0], verts[:, 1], verts[:, 2]                        # split the vertex table into three columns
    triang_v = mtri.Triangulation(E_v, N_v, t)                                   # convert to matplotlib's own triangle-mesh object, t = array of triangel corners numbers from above
    norm_v = mcolors.Normalize(vmin=np.nanpercentile(Z_v, 2),                    # Clip the colour scale at the 2nd/98th percentiles so a few extreme cells don't ruin scale
                               vmax=np.nanpercentile(Z_v, 98))

    fig, ax_main = plt.subplots(figsize=(12, 8))
    ax_main.tripcolor(triang_v, Z_v, cmap='terrain', shading='gouraud',
                      norm=norm_v, alpha=0.6)
    ax_main.triplot(triang_v, color='k', linewidth=0.2, alpha=0.4)
    ax_main.set_title(f'Terrain mesh — {n_v:,} vertices  {n_t:,} triangles\n'
                      f'STRIDE={mesh_stride}  SIMPLIFY={mesh_simplify}')
    ax_main.set_aspect('equal')
    ax_main.ticklabel_format(style='sci', axis='both', scilimits=(4, 4))
    ax_main.set_xlabel('Easting (m)')
    ax_main.set_ylabel('Northing (m)')

    # Inset over the central 15% of the extent, where triangles are big enough to see individually.
    zoom   = 0.15
    E_span = E_v.max() - E_v.min()
    N_span = N_v.max() - N_v.min()
    x0, x1 = E_v.mean() - E_span*zoom/2, E_v.mean() + E_span*zoom/2
    y0, y1 = N_v.mean() - N_span*zoom/2, N_v.mean() + N_span*zoom/2

    # inset_axes places a small plot inside the big one. The four numbers are [left, bottom, width, height] as fractions of the main axes.
    ax_inset = ax_main.inset_axes([0.62, 0.05, 0.36, 0.36])
    ax_inset.tripcolor(triang_v, Z_v, cmap='terrain', shading='gouraud',
                       norm=norm_v, alpha=0.6)
    ax_inset.triplot(triang_v, color='k', linewidth=0.5, alpha=0.7)
    ax_inset.set_xlim(x0, x1)
    ax_inset.set_ylim(y0, y1)
    ax_inset.set_aspect('equal')
    ax_inset.set_title(f'Central {int(zoom*100)}% zoom', fontsize=9)
    ax_inset.tick_params(labelsize=7)
    ax_inset.ticklabel_format(style='sci', axis='both', scilimits=(4, 4))

    # Draw the zoom box and its connecting lines on the main axes.
    ax_main.indicate_inset_zoom(ax_inset, edgecolor='red')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Cell 9 — Raycasting: depth image and mask
# ══════════════════════════════════════════════════════════════════════════════

def build_raycasting_scene(mesh_path):
    """Load the terrain mesh written by Cell 8 and build the Open3D raycasting scene."""
    # ──────────────────────────────────────────────────────────────────────────────
    # ── Load the mesh and check it makes sense next to the camera ─────────────────
    print(f'Loading mesh: {mesh_path}')
    mesh = o3d.io.read_triangle_mesh(mesh_path)     # read the .obj written by Cell 8
    #mesh.remove_duplicated_vertices()               # Optional: if .obj file from another GIS program, may need to remove duplicates.
    #mesh.remove_duplicated_triangles()

    # check what was loaded and report summary stats, including a check if number of vertices = 0.
    verts = np.asarray(mesh.vertices)
    n_v   = verts.shape[0]
    n_t   = np.asarray(mesh.triangles).shape[0]
    print(f'  {n_v:,} vertices   {n_t:,} triangles')
    if n_v == 0:
        raise RuntimeError('The mesh has no vertices. Check the .obj file.')


    ##---- Optional check for mesh extent, if depth image doesnt work ----------
    #print(f'  Vertex E: {verts[:,0].min():.1f} -> {verts[:,0].max():.1f}')
    #print(f'  Vertex N: {verts[:,1].min():.1f} -> {verts[:,1].max():.1f}')
    #print(f'  Vertex Z: {verts[:,2].min():.1f} -> {verts[:,2].max():.1f}')

    ## A camera thousands of kilometres from the mesh means the two are in different
    ## coordinate systems — almost always a CRS mismatch between DEM and GCPs.
    ## Find the approx. distance from the camera to the mesh
    #dist_cam_mesh = np.linalg.norm(np.array([cam_x, cam_y]) - verts[:, :2].mean(axis=0))        # verts[:, :2] keeps the easting and northing columns; mean(axis=0) averages them.
    #print(f'  Camera -> mesh centre: {dist_cam_mesh:.0f} m')
    #if dist_cam_mesh > 1e6:                                                                     # if more than 1000 km away, raise a warning
    #    print('  WARNING: camera and mesh are very far apart. Check that the GCP')
    #    print('  world coordinates and the DEM are in the same CRS.')


    # ── Build the raycasting scene ────────────────────────────────────────────────
    # Open3D builds a bounding volume hierarchy (BVH) over the triangles, an efficient way to filter ray hits
    # so each ray test costs log(n_triangles) rather than n_triangles.
    print('\nBuilding raycasting scene...')
    scene  = o3d.t.geometry.RaycastingScene()                   # an empty scene to put geometry into
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)      # convert to the newer "tensor" mesh type the scene needs (compatible with PyTorch)
    scene.add_triangles(mesh_t)                                 # hand over the triangles and build the BVH
    return scene


def cast_image_rays(scene, K, R, cam_world, w_img, h_img):
    """
    Fire one ray from the camera through every image pixel.

    Returns
    -------
    rays_world    : (H*W, 3) unit ray directions in world space
    depth_image   : (H, W) distance to the terrain in metres, 0 where nothing was hit
    coverage      : percentage of pixels that hit terrain
    camera_lifted : True if the ray origin had to be lifted above the mesh
    dZ            : the lift applied, in metres
    """
    # ── One ray direction per image pixel ─────────────────────────────────────────
    print(f'Building ray grid ({w_img} x {h_img} = {w_img*h_img:,} rays)...')

    # Build arrays of pixel indices
    cols = np.arange(w_img, dtype=np.float64)                 # 0, 1, 2, ... up to the image width
    rows = np.arange(h_img, dtype=np.float64)                 # 0, 1, 2, ... up to the image height
    col_grid, row_grid = np.meshgrid(cols, rows)              # pair every column with every row
    pts_img = np.stack([col_grid.ravel(), row_grid.ravel()], axis=1)   # flatten into an (H*W, 2) list of [col, row]

    # Calculate the view direction of each pixel using the camera model
    # Using undistortPoints inverts K (and lens distortion, if supplied), turning pixel
    # coordinates into normalised camera-space directions. Inverse of OpenCV's
    # convention guarantees the same convention as solvePnP.
    pts_norm = cv2.undistortPoints(pts_img.reshape(-1, 1, 2),                                   # reshaped to the (N,1,2) layout OpenCV wants
                                   cameraMatrix=K,                                              # the intrinsics from Cell 5
                                   distCoeffs=np.array([0.0, 0.0, 0.0, 0.0])).reshape(-1, 2)    # distortion coefficients, and flatten the result back

    # Normalise camera 'ray length units' to convert to 'metre units' by dividing by itself
    # A normalised point (x, y) corresponds to the camera-space direction (x, y, 1).
    rays_cam  = np.concatenate([pts_norm, np.ones((len(pts_norm), 1))], axis=1)             # glue a column of 1s on the right
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)                             # divide (by itself) each row by its length so all are unit vectors

    # Convert ray orientations from camera space to world space by transposing rotation matrix
    # R maps world -> camera, so R.T maps camera -> world.
    rays_world = (R.T @ rays_cam.T).T   # (H*W, 3) unit vectors in world space

    # ── Make sure the ray origin is above the terrain surface ─────────────────────
    # If the solved camera centre sits at or just below the mesh (common when the
    # camera stood on ground the DEM smooths over, or where the DEM has a flat base
    # plane at Z=0), every ray immediately hits the underside of a nearby triangle
    # and the depth image comes out empty.
    #
    # Test: cast one ray straight up. If it hits something within 50 m, the camera is
    # under the surface. Tricky solution is to lift the origin just above it.
    # Make test array:
    test_up = np.array([[cam_world[0], cam_world[1], cam_world[2],                      # A ray is six numbers: three for where it starts, three for which way it points.
                         0.0, 0.0, 1.0]], dtype=np.float32)                             # start at the camera, then point straight up (+Z)

    # cast_rays returns a dict; ['t_hit'] is an array of hit distances, one per ray. .numpy() converts it from Open3D's tensor type, and [0] takes the ray's distance as a plain number
    dist_up = scene.cast_rays(
        o3d.core.Tensor(test_up, dtype=o3d.core.Dtype.Float32))['t_hit'].numpy()[0]     # 't_hit' is Open3D's naming convention for distance

    # check if test ray hit the mesh directly above
    if np.isfinite(dist_up) and dist_up < 50.0:                                                 # if yes and distance less than 50m, correct camera position
        dZ         = float(dist_up) + 0.05                                                      # distance to surface + 5 cm clearance
        cam_origin = (cam_world + np.array([0, 0, dZ])).astype(np.float32)                      # same position, raised in Z by the amount needed
        camera_lifted = True                                                                    # record that the camera position was changed
        print(f'  Camera lifted {dZ:.3f} m in Z (mesh surface was {dist_up:.3f} m above it)')   # report
        if dZ > 2.0:                                                                            # warn if correction was very large >2m. if so probably a larger issue with GCPs or FOV
            print('  WARNING: lift exceeded 2 m. The solved camera height is well below')
            print('  the terrain, which suggests a GCP or FOV problem. A large lift can')
            print('  also push ridgelines into the sky mask and lose them.')
    else:                                                                                       # if no hit, then don't apply a correction
        dZ         = 0.0
        cam_origin = cam_world.astype(np.float32)
        camera_lifted = False
        print('  Camera origin is clear of the mesh — no lift needed.')

    # ── Cast every ray ────────────────────────────────────────────────────────────
    # First supply the origins of the rays in an efficient way (broadcasting)
    # Open3D wants rays as rows of [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z].
    # broadcast_to repeats the one origin for every ray without copying the memory.
    origins  = np.broadcast_to(cam_origin, (len(rays_world), 3))
    rays_o3d = np.concatenate([origins.astype(np.float32),               # the origin columns
                               rays_world.astype(np.float32)], axis=1)   # and the direction columns beside them

    # now calculate distance from origin to terrain
    print(f'Casting {len(rays_o3d):,} rays...')                                             # annouce
    result = scene.cast_rays(o3d.core.Tensor(rays_o3d, dtype=o3d.core.Dtype.Float32))       # cast the rays using open3d format
    dist_hit  = result['t_hit'].numpy()                                                     # distance from the (possibly lifted) origin

    # ── Convert distances back to the true camera position ────────────────────────
    # If the origin was lifted, t_hit is measured from the lifted point. Downstream
    # code reconstructs world positions as cam_world + ray * dist_hit, so dist_hit must
    # be the distance from the ORIGINAL camera. Computing it from the intersection
    # point gives that distance exactly, at any ray angle.
    #
    # Note that the reconstructed point then sits off the terrain surface by roughly
    # the lift distance, because it is measured along a ray from the unlifted camera.
    # Mostly irrelevant with a few centimetres lifted, but lift of a few metres introduces
    # error.
    if camera_lifted:                                                                       # if camera was lifted
        hit_mask = np.isfinite(dist_hit)                                                    # create a true/false 'hit mask'
        xyz_hits = cam_origin + rays_world * dist_hit[:, np.newaxis]                        # find where each ray actually hit using # [:, np.newaxis] turns the flat distance list into a column so each distance # multiplies all three components of its own direction
        dist_hit[hit_mask] = np.linalg.norm(xyz_hits[hit_mask] - cam_world, axis=1)         # re-measure from the true camera. Subset xyz_hits using hit_mask, minus cam_world which is the camera's real position [E, N, Z], giving the arrow from camera to hit. norm(axis=1) is its length

    # ── Create depth image: pixel value = distance, 0 where nothing was hit ──────────────────
    dist_hit_img   = dist_hit.reshape(h_img, w_img)                                             # reshape the flat list back into an image
    depth_image = np.where(np.isfinite(dist_hit_img), dist_hit_img, 0.0).astype(np.float64)     # infinity (a miss) becomes 0

    #helpful stats
    valid_count = (depth_image > 0).sum()                                                       # count the pixels that found terrain
    coverage    = valid_count / depth_image.size * 100                                          # .size is the total pixel count, x100 makes a percentage
    print(f'\nDepth image:')                                                                    # header
    print(f'  Pixels hitting terrain: {valid_count:,}  ({coverage:.1f}% of the image)')         # how much of the frame found terrain
    if valid_count > 0:                                                                         # if something was hit
        d = depth_image[depth_image > 0]                                                        # keep just the non-zero distances
        print(f'  Distance range: {d.min():.0f} -> {d.max():.0f} m')                            # nearest and furthest terrain

    # ── Optional: Diagnose an empty or sparse result ────────────────────────────────────────
    #if valid_count == 0:
    #    print('\nDepth image is empty — running a diagnostic ray...')
    #    # Aim one ray straight at the GCP centroid, which is definitely terrain.
    #    to_gcps  = (GCP_WORLD.mean(axis=0) - cam_world).astype(np.float32)             # vector from camera to the GCP centroid
    #   to_gcps /= np.linalg.norm(to_gcps)                                                  # scale it to unit length
    #    test_ray = np.array([[*cam_origin, *to_gcps]], dtype=np.float32)                   # The * prefix unpacks a list into separate values, so this builds one row of six numbers.
    #    test_t   = scene.cast_rays(                                                        # fire that single ray
    #        o3d.core.Tensor(test_ray, dtype=o3d.core.Dtype.Float32))['t_hit'].numpy()[0]
    #    if np.isfinite(test_t):                                                            # it hit, so the mesh is reachable
    #        print(f'  The mesh IS reachable ({test_t:.0f} m towards the GCPs), so the')    # the geometry reaches the mesh, so the pose is the problem
    #        print('  camera is pointing the wrong way — re-check the pose in Cell 5.')     # and Cell 5 is where to look
    #    else:                                                                              # even this ray missed
    #        print('  The mesh is not reachable even towards the GCPs. Check that the')     # the mesh cannot be reached at all
    #        print('  DEM covers the photographed area and rebuild in Cell 8.')             # so the DEM extent is the problem
    #elif coverage < 10:                                                                    # very little of the frame found terrain
    #    print(f'\nLow coverage ({coverage:.1f}%). If much of the photo is terrain, the')   # some terrain was found, but very little
    #    print('  DEM probably does not extend far enough — use a wider DEM.')              # the usual cause
    #else:                                             # a normal result
    #    print('\nDepth image looks valid.')                                                # the normal case

    return rays_world, depth_image, coverage, camera_lifted, dZ


def plot_depth_image(img_rgb, depth_image, coverage):
    """Show the depth image beside the photo."""
    # ── Show the depth image beside the photo ─────────────────────────────────────
    depth_disp = depth_image.copy()                                                     # .copy() so editing this does not change depth_image
    depth_disp[depth_disp == 0] = np.nan                                                # NaN renders blank, not dark blue
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))                                     # one row of two plots; axes is a list of them
    axes[0].imshow(img_rgb)                                                             # left panel: the photo
    axes[0].set_title('Original photo')                                                 # its title
    axes[0].axis('off')                                                                 # no axis numbers
    im = axes[1].imshow(depth_disp, cmap='viridis')                                     # right panel: distance as colour; keep the handle for the colour bar
    plt.colorbar(im, ax=axes[1], label='Ray distance (m)')                              # the scale bar beside it
    axes[1].set_title(f'Depth image ({coverage:.1f}% coverage)\n'                       # title line 1
                      f'blank = no mesh hit (sky, or terrain outside the DEM)')         # title line 2
    axes[1].axis('off')                                                                 # no axis numbers here either
    plt.tight_layout()
    plt.show()


def build_exclusion_mask(depth_image, mask_mode, custom_mask_path, w_img, h_img):
    """
    Build the mask of pixels to ignore downstream (True = excluded).

    mask_mode : 'geometric', 'geometric+custom' or 'custom_only'
    """
    # ── Build the exclusion mask ──────────────────────────────────────────────────
    # Everything downstream (ortho, line projection) skips pixels flagged here.

    # Build geometric mask
    geometric_sky = (depth_image == 0)   # True where the ray hit nothing

    #Build custom mask
    custom_mask_img = None                                                                      # stays None unless a custom mask is loaded
    if mask_mode in ('geometric+custom', 'custom_only'):                                        # if mask mode requires custom mask...
        if custom_mask_path is None:                                                            #   if path not set, raise an error
            raise ValueError(f'MASK_MODE="{mask_mode}" needs CUSTOM_MASK_PATH set in Cell 2.')
        cust_raw = cv2.imread(custom_mask_path, cv2.IMREAD_GRAYSCALE)                           #       otherwise read it as a single-channel image
        if cust_raw is None:                                                                    #   If read fails... (OpenCV returns None when the read fails),
            raise FileNotFoundError(f'Cannot load custom mask: {custom_mask_path}')             #       raise an error
        if cust_raw.shape != (h_img, w_img):                                                    #   if the mask does not match the photo shape exactly
            cust_raw = cv2.resize(cust_raw, (w_img, h_img), interpolation=cv2.INTER_NEAREST)    #       resize with nearest neighbour interpolation
            print(f'  Custom mask resized to {w_img} x {h_img}')                                #       and say so, in case it was unintended
        custom_mask_img = cust_raw > 0                                                          # Then create mask where any non-zero pixel means "exclude"
        n_custom = custom_mask_img.sum()                                                        # and count how many that is and report summary
        print(f'  Custom mask loaded: {n_custom:,} pixels '
              f'({n_custom / custom_mask_img.size * 100:.2f}%)')

    # create final mask
    if mask_mode == 'geometric':                                    # ray misses only
        exclusion_mask = geometric_sky
        print('Mask mode: geometric only')
    elif mask_mode == 'geometric+custom':                           # ray misses + custom mask (using '|' ('or') operator)
        exclusion_mask = geometric_sky | custom_mask_img
        print('Mask mode: geometric + custom')
    elif mask_mode == 'custom_only':                                # custom mask only
        exclusion_mask = custom_mask_img
        print('Mask mode: custom only (no geometric masking)')
    else:                                                           # mask mode not valid
        raise ValueError('MASK_MODE must be "geometric", "geometric+custom" or '
                         f'"custom_only". Got: "{mask_mode}"')


    #report summary
    n_excluded = exclusion_mask.sum()                                                           # how many pixels are masked
    n_terrain  = (~exclusion_mask).sum()                                                        # how many kept
    print(f'Exclusion mask: {n_terrain:,} usable pixels  |  {n_excluded:,} excluded '           # as a percentage
          f'({n_excluded / exclusion_mask.size * 100:.1f}%)')

    return exclusion_mask


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Line feature projection
# Cell 10 — Load and validate line features
# ══════════════════════════════════════════════════════════════════════════════

def load_features(features_csv, w_img, h_img):
    """
    Read the digitised line features (feature_name, col, row) and validate them.

    Returns
    -------
    features_valid : OrderedDict of feature name -> (N, 2) array of [col, row]
    """
    # ──────────────────────────────────────────────────────────────────────────────
    features_raw = OrderedDict()                 # create empty container for feature name -> list of [col, row]

    with open(features_csv, newline='') as f:    # open the CSV, closing it at the end of the block
        reader = csv.DictReader(f)               # reads each line into a dict keyed by the header names
        if reader.fieldnames is None:            # fail if fieldnames (header) is empty and raise error
            raise ValueError(f'{features_csv} appears to be empty.')   # stop here rather than failing further down

        # Column names must be exactly: feature_name, col, row
        # Surrounding spaces are tolerated; capitalisation is not. Because columns
        # are matched by name, the order they appear in the file does not matter.
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        # Every required column name that is missing from the header.
        missing = [c for c in ('feature_name', 'col', 'row')
                   if c not in reader.fieldnames]
        if missing:                                                    # a non-empty list counts as True
            raise ValueError(f'Feature CSV is missing columns {missing}. '
                             f'Found: {reader.fieldnames}')            # adjacent f-strings join into one message

        # ── Read the vertices, grouped by feature name in file order ──────────────
        for rec in reader:                                             # loop over the data rows; rec is one row as a dict
            if rec['feature_name'] is None or not rec['feature_name'].strip():
                continue                                               # skip blank lines
            name = rec['feature_name'].strip()                         # which feature this vertex belongs to
            col  = float(rec['col'])                                   # the pixel column, as a number
            row  = float(rec['row'])                                   # the pixel row
            # setdefault returns the list for this name, creating an empty one first
            # if the name has not been seen before; .append then adds this vertex.
            features_raw.setdefault(name, []).append([col, row])

    if not features_raw:                         # the header was fine but no data rows followed
        raise ValueError(f'No features read from {features_csv}.')     # nothing to project

    #── Validate ──────────────────────────────────────────────────────────────────
    print('\nFeatures loaded:')                                    # set up a table to print
    print(f'  {"Feature name":<28} {"Vertices":>8}  Status')
    print(f'  {"-"*56}')

    features_valid = OrderedDict()                                  # empty container for the valid features worth projecting
    for name, pts in features_raw.items():                          # .items() yields each name with its list of vertices
        pts_np = np.array(pts)                                      # convert the list of [col, row] pairs into an (N, 2) array
        n_oob = ((pts_np[:, 0] < 0) | (pts_np[:, 0] >= w_img) |     # count number out of bounds.
                 (pts_np[:, 1] < 0) | (pts_np[:, 1] >= h_img)).sum()
        if len(pts) < 2:                                            # if only a a single point is not a line, geometry wont work downstream
            status = 'skipped — a line needs at least 2 vertices'   # note it, and do not keep it
        elif n_oob > 0:                                             # if some vertices are off the frame
            status = f'WARNING: {n_oob} vertices outside the image'
            features_valid[name] = pts_np                           # kept, but flagged
        else:                                                       # otherwise everything is in order
            status = 'ok'
            features_valid[name] = pts_np                           # keep it
        print(f'  {name:<28} {len(pts):>8}  {status}')              # one table row per feature

    print(f'\n  Image size: {w_img} x {h_img} px')                  # for comparison with the coordinates above
    print(f'  Features to project: {len(features_valid)}')          # how many survived validation.

    return features_valid


def plot_features(img_rgb, features_valid):
    """
    Draw the digitised features on the photo.

    Returns the colours used, so Cell 12 can keep each feature's colour.
    """
    # ── Draw them on the photo ────────────────────────────────────────────────────
    # `colours` is reused by Cell 12 so a feature keeps the same colour throughout.
    # linspace picks evenly spaced positions along the Set1 colour map, one per feature.
    # max(..., 1) avoids asking for zero colours if nothing was loaded.
    colours = plt.cm.Set1(np.linspace(0, 1, max(len(features_valid), 1)))
    fig, ax = plt.subplots(figsize=(14, 9))      # one figure, 14 x 9 inches
    ax.imshow(img_rgb)                           # the photo as background
    # zip pairs each feature with its colour. The (name, pts) brackets unpack the
    # name/vertices pair that .items() produces.
    for (name, pts), colour in zip(features_valid.items(), colours):
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=colour, linewidth=1.5, markersize=4,      #  '-o' = a line with a dot per vertex
                label=f'{name} ({len(pts)} pts)')                                           # the legend entry
        ax.annotate(name, xy=(pts[0, 0], pts[0, 1]), xytext=(6, 6),                         # name the feature at its first vertex
                    textcoords='offset points', fontsize=8, color='white',                  # offset in points from that vertex
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))             # a dark rounded box behind the text
    ax.legend(loc='upper right', fontsize=9)                                                # draw the legend
    ax.set_title('Digitised features on the original photo\n'
                 'Each line should trace the feature it was digitised from')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    print('If lines look mirrored or transposed, check that col and row are not swapped in the CSV.')        # and the usual fix if the lines look wrong
    return colours


# ══════════════════════════════════════════════════════════════════════════════
# Cell 11 — Project features to world coordinates
# ══════════════════════════════════════════════════════════════════════════════

def project_lines_via_mesh(pts_px, cam_world, R, K, raycast_scene,
                           cam_lifted=False, lift_dz=0.0):
    """
    Turn image pixel positions into world (E, N, Z) by ray-mesh intersection.

    The same unprojection via cv2.undistortPoints as Cell 9,
    same camera lift handling, same distance correction. This keeps the projected lines
    consistent with the ortho.

    Parameters
    ----------
    pts_px        : (N, 2) array of [col, row] pixel coordinates
    cam_world     : (3,)   camera position in world coordinates (unlifted)
    R             : (3, 3) rotation matrix, world -> camera
    K             : (3, 3) camera intrinsic matrix
    raycast_scene : the Open3D RaycastingScene built in Cell 9
    cam_lifted    : True if Cell 9 lifted the ray origin
    lift_dz       : the lift applied, in metres

    Returns
    -------
    xyz : (N, 3) world coordinates, NaN where the ray missed
    hit : (N,)   bool, True where the ray hit the mesh
    """
    # ── Pixels -> world-space ray directions ──────────────────────────────────
    # undistortPoints inverts the intrinsic matrix, turning pixel positions into
    # directions in the camera's own frame.
    pts_norm = cv2.undistortPoints(
        pts_px.astype(np.float64).reshape(-1, 1, 2),                                    # the (N,1,2) layout OpenCV expects
        cameraMatrix=K, distCoeffs=np.array([0.0, 0.0, 0.0, 0.0])).reshape(-1, 2)       # distortion coefficients, and flatten back to (N,2)

    rays_cam  = np.concatenate([pts_norm, np.ones((len(pts_norm), 1))], axis=1)         # append a column of 1s: (x, y) -> (x, y, 1)
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)                         # scale each row to unit length
    rays_w    = (R.T @ rays_cam.T).T                                                    # R.T maps camera -> world

    # ── Same origin as Cell 9 ────────────────────────────────────────────────
    if cam_lifted and lift_dz > 0.0:                                                    # If Cell 9 raised the origin
        origin = (cam_world + np.array([0.0, 0.0, lift_dz])).astype(np.float32)         # add the lift to the Z component
    else:                                                                               #o therwise leave the same (well, convert to float32 for Open3D)
        origin = cam_world.astype(np.float32)

    # ── Cast ──────────────────────────────────────────────────────────────────
    origins  = np.broadcast_to(origin, (len(rays_w), 3))                                # repeat the one origin for every ray, without copying
    rays_o3d = np.concatenate([origins.astype(np.float32),                              # six columns per ray: 3 for origin...
                               rays_w.astype(np.float32)], axis=1)                      # ...then 3 for direction
    dist_hit = raycast_scene.cast_rays(                                                 # cast the rays and return the distance of the hit
        o3d.core.Tensor(rays_o3d, dtype=o3d.core.Dtype.Float32))['t_hit'].numpy()

    hit = np.isfinite(dist_hit)                                       #create true/false for hit or miss
    xyz = np.full((len(dist_hit), 3), np.nan)                         #create empty results table with NaN, so misses stay obviously empty

    if hit.any():                                                               # .any() is True if at least one ray hit something
        if cam_lifted and lift_dz > 0.0:                                        # if cam was lifted...
            xyz_hits   = origin + rays_w[hit] * dist_hit[hit, np.newaxis]       # find where rays actually hit in world positions
            dist_hit[hit] = np.linalg.norm(xyz_hits - cam_world, axis=1)        # recalcualte unlifted camera

        xyz[hit] = cam_world + rays_w[hit] * dist_hit[hit, np.newaxis]          #otherwise, just fill the world position of every hit

    return xyz, hit


def project_features(features_valid, cam_world, R, K, scene, camera_lifted=False, dZ=0.0):
    """
    Project every digitised feature onto the terrain.

    Returns
    -------
    features_world : dict of feature name -> {'xyz', 'hit', 'px'}
    """
    print('Projecting line features to world coordinates...\n')

    # ── Pick up the camera lift settings from Cell 9 ─────────────────────────────
    # camera_lifted and dZ are passed in from Cell 9.
    _cam_lifted = camera_lifted                                              # use Cell 9's value
    _lift_dz    = dZ                                                         # likewise for the lift distance
    if _cam_lifted:                                                          # only worth mentioning when a lift applies
        print(f'Applying the Cell 9 camera lift: dZ = {_lift_dz:.3f} m\n')   # report the lift so the two cells can be seen to agree


    # ── Project every feature ─────────────────────────────────────────────────────
    features_world = {}                                                         # create an empty dict. feature name -> its results

    # loop to call the function for each feature, collecting the positions, hit flags, and pixels together into features_world
    for name, pts_px in features_valid.items():                  # one pass per digitised feature
        xyz, hit = project_lines_via_mesh(                       # two returned arrays, unpacked into two names
            pts_px, cam_world, R, K, scene,              # the vertices, camera, rotation, intrinsics and scene
            cam_lifted=_cam_lifted, lift_dz=_lift_dz)               # named arguments, so their order does not matter

        features_world[name] = {'xyz': xyz, 'hit': hit, 'px': pts_px}   # store all three under this feature's name

        # Report any vertex that found no terrain.
        # np.where(~hit)[0] lists the positions where hit is False.
        for idx in np.where(~hit)[0]:
            print(f'  "{name}" vertex {idx+1}: col={pts_px[idx,0]:.0f} '    # idx+1 so the first vertex reads as 1, not 0
                  f'row={pts_px[idx,1]:.0f} — ray missed the mesh '
                  f'(sky, or beyond the DEM)')

    #print summary
    # sum(... for ...) is a generator expression: it totals a value computed per feature.
    n_hit_total  = sum(int(d['hit'].sum())  for d in features_world.values())           # vertices that found terrain
    n_miss_total = sum(int((~d['hit']).sum()) for d in features_world.values())         # vertices that did not
    print(f'\nProjected {len(features_world)} features: '
          f'{n_hit_total} vertices hit, {n_miss_total} missed.')
    print('Continue to Cell 12 to write the shapefile.')

    return features_world


# ══════════════════════════════════════════════════════════════════════════════
# Cell 12 — Write shapefile
# ══════════════════════════════════════════════════════════════════════════════

#function to help with gaps in lines
def xyz_to_linestrings(xyz, hit):
    """
    Split a vertex sequence into LineStrings at every missed vertex.

    Geometry is 2D (E, N) — elevation is dropped because most GIS tools handle
    2D lines more predictably, and Z is recoverable from the DEM anyway.
    """
    segments, current = [], []                              #empty container for the final lines and current segment
    for i, is_hit in enumerate(hit):                        # walk the vertices, i counting from 0
        if is_hit:                                          # this vertex found terrain (true/false)
            current.append((xyz[i, 0], xyz[i, 1]))          # add its easting and northing to the run
        else:                                               #otherwise it missed and run ends here
            if len(current) >= 2:                           # if more than two points (A gap ends the current run, and runs of one point are not a line)
                segments.append(LineString(current))        # turn the run into a shapely line
            current = []                                    # start a fresh, empty run

    if len(current) >= 2:                                   # the loop may end mid-run, so close the last one
        segments.append(LineString(current))                # close off the run that just ended
    return segments                                         # hand back the list of lines


def write_feature_shapefile(features_world, output_shp, out_crs, image_path, err, inlier_mask=None):
    """
    Write the projected features to a shapefile in the DEM's CRS.

    Returns the GeoDataFrame that was written.
    """
    # ── Configuration ─────────────────────────────────────────────────────────────
    #initial check
    if not GEOPANDAS_OK:
        raise ImportError('geopandas and shapely are required: pip install geopandas shapely')

    # Camera model error: mean over the RANSAC inliers (the GCPs the pose was solved
    # from), with the mean over all GCPs alongside for transparency.
    err_in = err[inlier_mask] if inlier_mask is not None else err

    # ── Build one record per feature ──────────────────────────────────────────────
    rows_out, skipped = [], []                                  # empty container for rows for the shapefile, and names that could not be used

    for name, data in features_world.items():                   # one pass per projected feature
        xyz, hit, px = data['xyz'], data['hit'], data['px']     # pull the three arrays out of the stored dict

        if hit.sum() < 2:                                                           # fewer than two vertices found terrain
            print(f'  Skipping "{name}" — fewer than 2 vertices hit the terrain.')  # explain why this feature is being left out
            skipped.append(name)                                                    # remember it for the report
            continue                                                                # move to the next feature

        segments = xyz_to_linestrings(xyz, hit)                                     # split the vertices into one or more lines
        if not segments:                                                            # an empty list: no run was long enough
            skipped.append(name)                                                    # record the name for the summary
            continue                                                                # and move on to the next feature

        # One run -> LineString.
        # Several runs -> MultiLineString.
        geom = segments[0] if len(segments) == 1 else MultiLineString(segments)

        #build rows dict
        rows_out.append({                                 # a dict per shapefile row where the keys become the columns
            'geometry':   geom,                           # the line itself
            'name':       name,                           # the feature name from the CSV
            'img_src':    os.path.basename(image_path),   # source
            'n_verts':    int(len(px)),                   # how many vertices were digitised
            'n_hit':      int(hit.sum()),                 # how many found terrain
            'n_miss':     int((~hit).sum()),              # how many did not
            'reproj_err': round(float(err_in.mean()), 3), # camera model quality (inlier GCPs)
            'reproj_all': round(float(err.mean()), 3),    # the same over all GCPs, incl. RANSAC outliers
        })

    # if every feature was skipped print warning
    if not rows_out:
        raise RuntimeError('Nothing projected — check that the DEM covers the area '
                           'the features were digitised in.')

    # build a geopandas table from the rows, tagged with the CRS
    # write to shapefile
    gdf = gpd.GeoDataFrame(rows_out, crs=out_crs)
    gdf.to_file(output_shp)

    #report results
    print(f'Shapefile written: {os.path.abspath(output_shp)}')
    print(f'  CRS      : {out_crs}')
    print(f'  Features : {len(rows_out)}')
    if skipped:
        print(f'  Skipped  : {skipped}')
    print()
    print(f'  {"Feature":<28} {"Vertices":>8}  {"Hit":>5}  {"Miss":>5}')
    print(f'  {"-"*52}')
    for row in rows_out:
        print(f'  {row["name"]:<28} {row["n_verts"]:>8}  {row["n_hit"]:>5}  {row["n_miss"]:>5}')

    print(f'\n{os.path.basename(output_shp)} is ready for GIS. Cell 13 adds '
          f'uncertainty attributes.')
    return gdf


def plot_projected_features(img_rgb, features_world, colours, dem):
    """Plot the features on the photo beside where they landed on a DEM hillshade."""
    dem_data, dem_res_x   = dem['dem_data'], dem['dem_res_x']
    dem_bounds, out_crs   = dem['dem_bounds'], dem['out_crs']

    # ── Check plot: photo beside DEM hillshade ────────────────────────────────────
    # Hillshade uses the DEM from Cell 7, so this check is independent of the ortho.
    # try/except since the plot is optional, so a failure here will not stop the whole cell.
    try:
        from matplotlib.colors import LightSource

        ls = LightSource(azdeg=315, altdeg=45)                                  # light from the NW, 45 deg up
        hs = ls.hillshade(dem_data, vert_exag=2, dx=dem_res_x, dy=dem_res_x)    # shaded relief, vert_exag exaggerates height

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Left: what was digitised.
        axes[0].imshow(img_rgb)                                             # the photo as background
        for (name, data), colour in zip(features_world.items(), colours):   # same colours as Cell 10
            if data['hit'].sum() < 2:                                       # skip features that were not written
                continue
            px = data['px'][data['hit']]                                    # keep only the vertices that found terrain
            axes[0].plot(px[:, 0], px[:, 1], '-o', color=colour,            # draw them in image coordinates
                         linewidth=2, markersize=4, label=name)
        axes[0].set_title('Digitised pixels on the original photo')
        axes[0].legend(loc='upper right', fontsize=9)                       # legend from the labels above
        axes[0].axis('off')                                                 # no axis numbers

        # Right: where it landed on the ground.
        axes[1].imshow(hs, cmap='grey',                                     # the hillshade in greyscale
                       extent=[dem_bounds.left, dem_bounds.right,           # extent tells matplotlib the real-world
                               dem_bounds.bottom, dem_bounds.top],          # coordinates of the image corners
                       origin='upper', alpha=0.8)                           # row 0 is the top of the raster
        for (name, data), colour in zip(features_world.items(), colours):   # the same features again
            if data['hit'].sum() < 2:                                       # skip the ones not written
                continue
            xyz_hit = data['xyz'][data['hit']]                              # world coordinates of the vertices that hit
            axes[1].plot(xyz_hit[:, 0], xyz_hit[:, 1], '-o', color=colour,  # easting against northing
                         linewidth=2, markersize=4, label=name)
        axes[1].set_xlabel('Easting (m)')
        axes[1].set_ylabel('Northing (m)')
        axes[1].set_title('Projected lines on the DEM hillshade')
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].ticklabel_format(style='sci', axis='both', scilimits=(4, 4))   # scientific notation for large UTM values

        plt.suptitle(f'Line features — {out_crs}', fontsize=10)
        plt.tight_layout()
        plt.show()

    except Exception as e:                                           # any plotting failure lands here
        print(f'  (Check plot skipped: {e})')                        # report it, but keep the shapefile


# ══════════════════════════════════════════════════════════════════════════════
# Cell 13 — Line uncertainty
# ══════════════════════════════════════════════════════════════════════════════

# Function to re-solve the camera at a different FOV, used to turn the FOV
# uncertainty from Cell 4 into metres on the ground.
def camera_at_fov(fov_deg, gcp_pixel, gcp_world, w_img, h_img, ransac_threshold=20.0):
    """
    Solve the camera at a given FOV exactly as Cell 5 does, without printing.

    Returns R (world -> camera rotation), cam_world (camera position) and K.
    """
    focal_f = fov_to_focal_px(fov_deg, h_img)                                   # focal length (px) at this FOV
    K_f = np.array([[focal_f, 0,       w_img / 2.0 - 0.5],                      # same intrinsic matrix layout as Cell 5
                    [0,       focal_f, h_img / 2.0 - 0.5],
                    [0,       0,       1.0              ]], dtype=np.float64)
    world_centre_f = gcp_world.mean(axis=0)                                     # centre the coordinates, as in Cell 5
    world_c = (gcp_world - world_centre_f).reshape(-1, 1, 3).astype(np.float64)
    pixel   = gcp_pixel.reshape(-1, 1, 2).astype(np.float64)
    zeros4  = np.array([0.0, 0.0, 0.0, 0.0])                                    # no lens distortion, as in Cell 5

    # Same RANSAC solve as Cell 5
    success, rvec, tvec, _ = cv2.solvePnPRansac(world_c, pixel, K_f, zeros4,
                                                iterationsCount=10000, reprojectionError=ransac_threshold,
                                                confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:                                                              # no plain-solve fallback, as in Cell 5
        raise RuntimeError(f'Could not solve the camera at FOV = {fov_deg:.2f} deg: fewer than '
                           f'5 GCPs agree within {ransac_threshold} px.')

    R_f   = cv2.Rodrigues(rvec)[0]                                              # rotation vector -> 3x3 matrix
    cam_f = (-R_f.T @ tvec).ravel() + world_centre_f                            # camera centre, back in the DEM's CRS
    return R_f, cam_f, K_f


# Function to report whether a camera sits under the terrain surface (for information:
# the FOV term below does not depend on it).
def depth_below_surface(raycast_scene, cam):
    """
    Cast one ray straight up from the camera. A DEM mesh has no overhangs, so any
    surface directly above means the camera is underground: return how far below
    the surface it is, in metres. Otherwise return 0.
    """
    test_up = np.array([[cam[0], cam[1], cam[2], 0.0, 0.0, 1.0]], dtype=np.float32)   # start at the camera, point straight up (+Z)
    dist_up = raycast_scene.cast_rays(
        o3d.core.Tensor(test_up, dtype=o3d.core.Dtype.Float32))['t_hit'].numpy()[0]  # distance to the surface above, inf if none
    return float(dist_up) if np.isfinite(dist_up) else 0.0


def fov_perturbed_cameras(fov, fov_sigma_deg, gcp_pixel, gcp_world, w_img, h_img, ransac_threshold=20.0,
                          raycast_scene=None):
    """
    Solve the camera at FOV - sigma, FOV and FOV + sigma (sigma from Cell 4).

    A different FOV does not just change the lens: the pose solve re-fits the camera
    position and direction to the GCPs, so the two move together. Casting the lines
    through these cameras (in compute_uncertainty_profile) measures the combined effect
    on the ground directly, rather than approximating it with a formula.

    A camera standing close to the ground can end up below it, or behind a rise, when
    re-solved at a different FOV. If raycast_scene is given, that is reported; the FOV
    term handles it by measuring where each ray crosses the terrain nearest the feature
    (see project_lines_near_point), so it does not affect the result.

    Returns a dict {'nominal', 'minus', 'plus'} of (R, cam_world, K), or None if
    fov_sigma_deg is 0 or None (FOV treated as exact).
    """
    if not fov_sigma_deg or fov_sigma_deg <= 0:                                 # no FOV uncertainty to propagate
        print('FOV uncertainty is 0 — no FOV term in the line uncertainty.')
        return None

    try:
        cams = {'nominal': camera_at_fov(fov,                 gcp_pixel, gcp_world, w_img, h_img, ransac_threshold),
                'minus':   camera_at_fov(fov - fov_sigma_deg, gcp_pixel, gcp_world, w_img, h_img, ransac_threshold),
                'plus':    camera_at_fov(fov + fov_sigma_deg, gcp_pixel, gcp_world, w_img, h_img, ransac_threshold)}
    except RuntimeError as e:                                                   # one of the three could not be solved
        print(f'WARNING: {e}')
        print('  sigma_FOV is NOT included in the line uncertainty. Raise RANSAC_THRESHOLD, or')
        print('  reduce the FOV uncertainty, and re-run.')
        return None

    # Report how far the solved camera moves - a quick feel for how much the FOV matters
    move_m = np.linalg.norm(cams['minus'][1] - cams['nominal'][1])
    move_p = np.linalg.norm(cams['plus'][1]  - cams['nominal'][1])
    print(f'FOV {fov:.2f} +/- {fov_sigma_deg:.2f} deg: the solved camera moves '
          f'{move_m:.1f} m (FOV - sigma) and {move_p:.1f} m (FOV + sigma)')
    if raycast_scene is not None:                                               # report any camera below the surface
        for key, label in (('minus', 'FOV - sigma'), ('plus', 'FOV + sigma')):
            depth = depth_below_surface(raycast_scene, cams[key][1])
            if depth > 0:
                print(f'  The {label} camera sits {depth:.1f} m below the terrain surface '
                      f'(handled: rays are matched to the terrain nearest each feature)')
    return cams


# Function to find where a camera's rays cross the terrain NEAREST a reference point,
# rather than where they first meet it. Used for the FOV +/- sigma cameras: the ray
# is meant to reach the feature, so terrain beside the camera (the ground surface if
# the camera is under it, or a rise it stands behind) must not intercept it.
def project_lines_near_point(pts_px, cam_world, R, K, raycast_scene, xyz_ref):
    """
    For each pixel, the terrain crossing along its ray that lies nearest xyz_ref
    (the point the best-FOV camera found for the same pixel).

    Method: take the point on the ray closest to xyz_ref, cast from there both
    forwards and backwards (but not back past the camera), and keep whichever
    terrain crossing is nearer. Returns (N, 3) E, N, Z - NaN where the ray finds
    no terrain or xyz_ref is NaN - and the matching hit flags.
    """
    # Pixels -> world-space ray directions (same three steps as project_lines_via_mesh)
    pts_norm = cv2.undistortPoints(
        pts_px.astype(np.float64).reshape(-1, 1, 2),
        cameraMatrix=K, distCoeffs=np.array([0.0, 0.0, 0.0, 0.0])).reshape(-1, 2)
    rays_cam  = np.concatenate([pts_norm, np.ones((len(pts_norm), 1))], axis=1)
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    rays_w    = (R.T @ rays_cam.T).T

    xyz = np.full((len(pts_px), 3), np.nan)                                     # NaN until a crossing is found
    ok  = np.all(np.isfinite(xyz_ref), axis=1)                                  # only pixels the best-FOV camera placed
    if not ok.any():
        return xyz, np.zeros(len(pts_px), dtype=bool)
    rays = rays_w[ok]

    # The point on each ray closest to the reference point (not behind the camera)
    s_near = np.maximum(np.sum((xyz_ref[ok] - cam_world) * rays, axis=1), 0.0)  # distance along the ray
    p_near = cam_world + rays * s_near[:, np.newaxis]

    # Cast forwards and backwards from there
    def _cast(starts, dirs):
        r = np.concatenate([starts, dirs], axis=1).astype(np.float32)          # six columns: start point, then direction
        return raycast_scene.cast_rays(o3d.core.Tensor(r, dtype=o3d.core.Dtype.Float32))['t_hit'].numpy().astype(np.float64)
    t_fwd = _cast(p_near,  rays)
    t_bwd = _cast(p_near, -rays)
    t_bwd[t_bwd > s_near] = np.inf                                              # a crossing behind the camera does not count

    # Keep the nearer crossing: forwards (+t) or backwards (-t) along the ray
    use_fwd = t_fwd <= t_bwd
    t_best  = np.where(use_fwd, t_fwd, -t_bwd)
    found   = np.isfinite(t_best)
    idx = np.where(ok)[0][found]
    xyz[idx] = p_near[found] + rays[found] * t_best[found, np.newaxis]
    hit = np.zeros(len(pts_px), dtype=bool); hit[idx] = True
    return xyz, hit


# Function to resample the line so uncertainty of each pixel is calculated
# because otherwise only clicked vertices would inform error calculations, and
# few vertices vs many vertices on same line would give different error estimates.
def interpolate_line_pixels(pts_px, spacing=1.0):
    """
    Resample a polyline at fixed pixel intervals along its path in image space.

    Decouples the uncertainty estimate from how densely the line was clicked.
    """
    cols_out, rows_out = [], []                                                     # container for collected pieces of the resampled path
    for i in range(len(pts_px) - 1):                                                # each segment runs from vertex i to vertex i+1
        c0, r0 = pts_px[i]                                                          # start of this segment: column and row
        c1, r1 = pts_px[i + 1]                                                      # end of it
        seg_len = np.hypot(c1 - c0, r1 - r0)                                        # hypot is Pythagoras: the segment's length in pixels
        if seg_len < 1e-6:                                                          # if two vertices in the same place, nothing to resample
            continue
        n_steps = max(int(np.ceil(seg_len / spacing)), 2)                           # one sample per pixel of length, at least 2
        endpoint = (i == len(pts_px) - 2)                                           # Drop the end point on every segment but the last, so shared vertices are not counted twice. # True only on the final segment
        cols_out.append(np.linspace(c0, c1, n_steps, endpoint=endpoint))            # evenly spaced columns along the segment
        rows_out.append(np.linspace(r0, r1, n_steps, endpoint=endpoint))            # and the matching rows
    if not cols_out:                                                                # if every segment was a duplicate
        return pts_px                                                               # hand back the original vertices unchanged
    return np.stack([np.concatenate(cols_out), np.concatenate(rows_out)], axis=1)   # concatenate joins the per-segment pieces. stack pairs them into (N, 2) rows


# Function to calculate the uncertainty of each pixel
def compute_uncertainty_profile(pts_px, cam_world, R, K, raycast_scene, reproj_px, digitise_px,
                                include_dem_error=False, dem_vert_err=0.0,
                                cam_lifted=False, lift_dz=0.0, fov_cams=None):
    """
    Per-pixel uncertainty along one line feature.

     Every error source here starts as an angle and becomes a distance when it is
     projected onto the ground:

    1 pixel of error  ->  an angle of 1/focal radians
    at range dist        ->  dist/focal metres on the ground

    So uncertainty grows linearly with distance from the camera. That is why the
    far end of an oblique line is always less certain than the near end, and why
    the per-point range (dist_m below) is the quantity everything is multiplied by.

    Two angular sources are always included:
      - the camera model's own misfit, measured by the GCP reprojection error
      - how precisely the line was clicked
    They are independent, so they combine in quadrature (root of sum of squares).

    A third term covers the FOV uncertainty from Cell 4 (when fov_cams is given):
    the line is re-cast through the cameras solved at FOV - sigma and FOV + sigma,
    and how far each point moves on the ground is its sigma_fov. It joins the other
    two in quadrature.

    A third, optional source is vertical rather than angular. if the DEM surface
    is wrong by dz, the ray meets it in the wrong place, displaced horizontally.

    raycast_scene, cam_lifted and lift_dz come from Cell 9; fov_cams from
    fov_perturbed_cameras.

    Returns a dict of sigma arrays (metres) plus the interpolated positions they
    correspond to.

    """

    # Step 1 - resample/interpolate line to get each pixel
    interp_px = interpolate_line_pixels(pts_px, spacing=1.0)
    focal = K[0, 0]                                                                         # the focal length sits at row 0, column 0 of K. Needed below for calculations

    # # Step 2 — rebuild each pixel's viewing direction in world coordinates.
    # Only needed for the optional DEM term, which depends on how steeply the
    # ray meets the ground.
    # Same three operations as Cell 9: invert camera matrix K, add the z=1 column, rotate into world axes.
    pts_norm = cv2.undistortPoints(
        interp_px.reshape(-1, 1, 2).astype(np.float64),
        cameraMatrix=K, distCoeffs=np.array([0.0, 0.0, 0.0, 0.0])).reshape(-1, 2)
    rays_cam  = np.concatenate([pts_norm, np.ones((len(pts_norm), 1))], axis=1)
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    rays_w    = (R.T @ rays_cam.T).T

    # Step 3 — find where each sample lands on the terrain, so its range from
    # the camera can be measured. Same raycasting as Cell 11, including potential lift corrections
    xyz_interp, hit_interp = project_lines_via_mesh(
        interp_px, cam_world, R, K, raycast_scene,
        cam_lifted=cam_lifted, lift_dz=lift_dz)

    # Step 4 — Range from the camera to each point in m
    dist_m = np.full(len(interp_px), np.nan)                                                       # start as NaN so misses stay blank
    dist_m[hit_interp] = np.linalg.norm(xyz_interp[hit_interp] - cam_world, axis=1)                # distance for every point that hit

    # Step 5 - Convert every angular error into metres on the ground at distance dist_m
    # Angular error (px / focal) x range = ground error.
    sigma_camera   = (reproj_px   / focal) * dist_m                                 # metres of error from the camera model
    sigma_digitise = (digitise_px / focal) * dist_m                                 # metres of error from clicking accuracy

    # Step 5b — the FOV uncertainty from Cell 4, as metres on the ground.
    # Cast the same samples through the cameras solved at FOV - sigma and FOV + sigma
    # and measure how far each lands (E, N) from where the best-FOV camera puts it.
    # The RMS of the two shifts is sigma_fov.
    #
    # The FOV +/- sigma rays are matched to the terrain crossing NEAREST the best-FOV
    # point (project_lines_near_point), not the first thing they meet. A camera standing
    # near the ground can end up below it, or behind a small rise, when re-solved; its
    # rays would then all hit the terrain beside it and sigma_fov would just equal each
    # point's distance from the camera.
    #
    # If only one of the two rays finds terrain, that one is used alone; if neither
    # does, the sample gets NaN and drops out of the summary, like other misses.
    # Both counts are reported by line_uncertainty.
    fov_unmatched = fov_one_side = fov_flagged = 0
    if fov_cams is not None:
        R_n, cam_n, K_n = fov_cams['nominal']
        xyz_nom, _ = project_lines_via_mesh(interp_px, cam_n, R_n, K_n, raycast_scene,
                                            cam_lifted=cam_lifted, lift_dz=lift_dz)       # best-FOV camera (same as Step 3)
        xyz_fov = {}
        for key in ('minus', 'plus'):
            R_k, cam_k, K_k = fov_cams[key]
            xyz_fov[key], _ = project_lines_near_point(interp_px, cam_k, R_k, K_k, raycast_scene, xyz_nom)
        d_minus = np.hypot(*(xyz_fov['minus'][:, :2] - xyz_nom[:, :2]).T)   # ground shift at FOV - sigma
        d_plus  = np.hypot(*(xyz_fov['plus'][:, :2]  - xyz_nom[:, :2]).T)   # ground shift at FOV + sigma

        # RMS of the two. If only one of them found terrain (the other camera's ray passes
        # under or over the whole surface - possible for features outside the GCP spread),
        # use that one alone; NaN only if neither did.
        d_sq = np.stack([d_minus**2, d_plus**2])
        n_ok = np.isfinite(d_sq).sum(axis=0)
        sigma_fov = np.full(len(interp_px), np.nan)
        sigma_fov[n_ok > 0] = np.sqrt(np.nansum(d_sq[:, n_ok > 0], axis=0) / n_ok[n_ok > 0])

        # Book-keeping for the report: samples where one or both FOV +/- sigma rays found
        # no terrain, and samples where sigma_fov is more than 10% of the range (worth a look)
        fov_one_side  = int((hit_interp & (n_ok == 1)).sum())
        fov_unmatched = int((hit_interp & (n_ok == 0)).sum())
        with np.errstate(invalid='ignore'):
            fov_flagged = int((sigma_fov > 0.1 * dist_m).sum())
    else:
        sigma_fov = np.zeros(len(interp_px))                                            # FOV treated as exact

    sigma_relative = np.sqrt(sigma_camera**2 + sigma_digitise**2 + sigma_fov**2)    # Independent errors add in quadrature.

    # Step 6 (optional) — the vertical error of the DEM.
    if include_dem_error and dem_vert_err > 0:                                           #if dem was meant to be included
        # View zenith = angle between the ray and vertical. A vertical error of
        # dz displaces the intersection horizontally by dz * tan(zenith), which
        # blows up for near horizontal rays — hence the cap just short of 90 degrees.
        rays_n = rays_w / np.maximum(                                                    # maximum() guards against dividing by zero by flooring the length at 1e-10
            np.linalg.norm(rays_w, axis=1, keepdims=True), 1e-10)
        zenith = np.arccos(np.clip(np.abs(rays_n[:, 2]), 0, 1))                          # arccos of the vertical component = the angle
        sigma_dem      = dem_vert_err * np.tan(np.clip(zenith, 0, np.radians(89.5)))     # the horizontal shift it causes
        sigma_absolute = np.sqrt(sigma_relative**2 + sigma_dem**2)                       # all three terms added in quadrature
    else:                                                                                # otherwise, continue in relative mode
        sigma_dem      = np.zeros_like(sigma_relative)                                   # zeros_like copies the shape, filled with 0
        sigma_absolute = sigma_relative.copy()                                           # absolute equals relative when the DEM is treated as truth

    # Return every term separately as well as the totals, so you can see
    # which source dominates rather than only the combined figure.
    return {'sigma_camera':   sigma_camera,
            'sigma_digitise': sigma_digitise,
            'sigma_fov':      sigma_fov,
            'fov_unmatched':  fov_unmatched,
            'fov_one_side':   fov_one_side,
            'fov_flagged':    fov_flagged,
            'sigma_relative': sigma_relative,
            'sigma_dem':      sigma_dem,
            'sigma_absolute': sigma_absolute,
            'xyz':            xyz_interp,
            'hit':            hit_interp,
            'interp_px':      interp_px}


# custom function to summarise the results
def summarise(d, include_dem=False):
    """
    Reduce a per-pixel profile to the numbers that go in the shapefile.
    """

    # Drop the samples that cannot contribute, then keep only the valid part of
    # each array so every statistic below is computed over the same points.
    valid = d['hit'] & np.isfinite(d['sigma_relative'])                 # pixels that hit terrain and produced a real number
    if valid.sum() < 2:                                                 # if too few to say anything about, return none
        return None
    sr, sc = d['sigma_relative'][valid], d['sigma_camera'][valid]       # keep only the valid samples
    sd, sa = d['sigma_digitise'][valid], d['sigma_absolute'][valid]
    sdm    = d['sigma_dem'][valid]
    sf     = d['sigma_fov'][valid]

    # Calculate gradient of uncertainty along the line, using the line length and min and max relative uncertainty
    xyz_v    = d['xyz'][valid]                                                          # world positions of the valid samples
    line_len = float(np.sum(np.hypot(np.diff(xyz_v[:, 0]), np.diff(xyz_v[:, 1]))))      # diff() gives the step between consecutive samples, hypot turns each pair of steps into a distance, sum() adds them into the total line length
    grad     = float((sr.max() - sr.min()) / line_len) if line_len > 0 else 0.0         # rise in uncertainty per metre (unless line length is 0)

    # what to return out of the summary
    out = {'sigma_camera_mean':   float(sc.mean()),       # average camera-model error along the line
           'sigma_digitise_mean': float(sd.mean()),       # average digitising error
           'sigma_fov_mean':      float(sf.mean()),       # average FOV-uncertainty error
           'sigma_relative_mean': float(sr.mean()),       # average of the camera, digitising and FOV terms combined
           'sigma_relative_min':  float(sr.min()),        # best point on the line (nearest the camera)
           'sigma_relative_max':  float(sr.max()),        # worst point (furthest away)
           'sigma_grad':          grad,                   # how fast it grows along the line
           'depth_varying':       bool(sr.max() > 2.0 * sr.min()),  # True if far end uncertainty more than twice the near end = the line runs away from the camera and a single mean is misleading.
           'n_px_samples':        int(valid.sum())}       # how many samples went into these numbers
    if include_dem:                                       # if absolute unncertainty needed, report these values too...
        out['sigma_dem_mean']      = float(sdm.mean())
        out['sigma_absolute_mean'] = float(sa.mean())
    return out


def line_uncertainty(features_world, cam_world, R, K, scene, err, focal_px,
                     digitise_px=1.0, include_dem_error=False, dem_vertical_error_m=10.0,
                     camera_lifted=False, dZ=0.0, fov_cams=None, fov_sigma_deg=0.0,
                     inlier_mask=None):
    """
    Estimate the uncertainty of every projected feature and print the results table.

    Returns
    -------
    uncertainty_results : dict of feature name -> summary dict (see summarise)
    reproj_px           : mean GCP reprojection error used, in pixels
    """
    # ──────────────────────────────────────────────────────────────────────────────
    # Gather the inputs and print a report here
    # The "1 px at 1000 m" line is the scale factor everything below depends on —
    # check it looks sensible before trusting the results table.

    # Camera model error from Cell 5: mean over the RANSAC inliers, the GCPs the pose
    # was solved from. Outliers did not shape the camera, so they do not count towards
    # its error. The all-GCP mean is printed alongside for transparency.
    err_in    = err[inlier_mask] if inlier_mask is not None else err
    REPROJ_PX = float(err_in.mean())   # camera model error from Cell 5

    print('Uncertainty inputs:')
    print(f'  Camera reprojection error : {REPROJ_PX:.2f} px (mean over {len(err_in)} inlier GCPs; '
          f'all {len(err)} GCPs: {err.mean():.2f} px)')
    print(f'  Digitising uncertainty    : {digitise_px:.1f} px')
    print(f'  FOV uncertainty           : '
          + (f'+/-{fov_sigma_deg:.2f} deg (from Cell 4)' if fov_cams is not None
             else 'not included (FOV treated as exact)'))
    print(f'  Focal length              : {focal_px:.1f} px')
    print(f'  1 px at 1000 m range      : {1000/focal_px:.2f} m on the ground')
    print(f'  DEM error                 : '
          + (f'{dem_vertical_error_m:.1f} m (absolute mode)' if include_dem_error
             else 'not included (relative mode)'))
    print()

    # ── Calculate uncertainty for every feature ─────────────────────────────────────────────────────
    # Set up container for outputs
    uncertainty_results = {}

    # Check, calculate and summarise uncertaint for each feature
    for name, data in features_world.items():                                  # one pass per projected feature
        pts_px, hit = data['px'], data['hit']                                  # filter only its digitised vertices and which of them hit terrain
        if hit.sum() < 2:                                                      # if too few hits to form a line, skip and move on
            print(f'  Skipping "{name}" — fewer than 2 terrain hits')
            continue

        # Only the vertices that hit are used, so a gap does not drag a straight
        # interpolated segment across untouched ground.
        d = compute_uncertainty_profile(                                                #using function above
            pts_px[hit], cam_world, R, K, scene, REPROJ_PX, digitise_px,
            include_dem_error=include_dem_error, dem_vert_err=dem_vertical_error_m,
            cam_lifted=camera_lifted, lift_dz=dZ, fov_cams=fov_cams)

        # Report FOV samples that could not be matched, or look implausibly large
        if d['fov_one_side']:
            print(f'  "{name}": at {d["fov_one_side"]} of {len(d["hit"])} samples only one of the '
                  f'FOV +/- sigma rays found terrain — sigma_FOV uses that one alone')
        if d['fov_unmatched']:
            print(f'  "{name}": {d["fov_unmatched"]} of {len(d["hit"])} samples dropped — '
                  f'neither FOV +/- sigma ray found terrain')
        if d['fov_flagged']:
            print(f'  WARNING: "{name}": sigma_FOV exceeds 10% of the range at {d["fov_flagged"]} '
                  f'samples — the FOV uncertainty moves it a long way (usually a feature outside '
                  f'the GCP spread, e.g. foreground); check the Cell 4 FOV uncertainty')

        #summarise results
        stats = summarise(d, include_dem=include_dem_error)
        if stats is None:                                                               #or skip and carry on if no results
            print(f'  Skipping "{name}" — not enough valid samples')
            continue

        # store the summary for the table and the shapefile
        uncertainty_results[name] = stats

    # ── Results table ─────────────────────────────────────────────────────────────
    # Table header anbd formatting
    # Build the header as a string first, so the rule below can be measured to match
    print('Per-feature uncertainty (metres, sampled at 1 px intervals)\n')
    hdr = (f'  {"Feature":<26}  {"s_camera":>9}  {"s_digitise":>10}  {"s_fov":>8}  '
           f'{"s_rel_mean":>10}  {"s_rel_min":>9}  {"s_rel_max":>9}  {"s_grad":>8}  Type')
    if include_dem_error:
        hdr += f'  {"s_DEM":>7}  {"s_abs_mean":>10}'
    print(hdr)
    print(f'  {"-"*(len(hdr)-2)}')

    # Print one row per feature in results, assembled the same way as the header
    for name, s in uncertainty_results.items():
        ltype = 'depth-varying' if s['depth_varying'] else 'front-on'
        row = (f'  {name:<26}  {s["sigma_camera_mean"]:>8.2f}m  '
               f'{s["sigma_digitise_mean"]:>9.2f}m  {s["sigma_fov_mean"]:>7.2f}m  {s["sigma_relative_mean"]:>9.2f}m  '
               f'{s["sigma_relative_min"]:>8.2f}m  {s["sigma_relative_max"]:>8.2f}m  '
               f'{s["sigma_grad"]:>8.5f}  {ltype}')
        if include_dem_error:
            row += f'  {s["sigma_dem_mean"]:>6.2f}m  {s["sigma_absolute_mean"]:>9.2f}m'
        print(row)

    # Explainer for the column abbreviations above
    print()
    print('  s_camera   : from the camera model (GCP reprojection error)')
    print('  s_digitise : from clicking accuracy')
    print('  s_fov      : from the FOV uncertainty (camera re-solved at FOV +/- sigma)')
    print('  s_rel_mean : camera, digitising and FOV terms combined, averaged along the line')
    print('  s_rel_min  : at the nearest point (best case)')
    print('  s_rel_max  : at the furthest point (worst case)')
    print('  s_grad     : growth in uncertainty per metre along the line')
    print('  Type       : depth-varying when s_rel_max > 2 x s_rel_min')
    if include_dem_error:
        print(f'  s_DEM      : DEM-induced horizontal error '
              f'({dem_vertical_error_m} m x tan(zenith))')
        print('  s_abs_mean : everything combined, vs true ground')

    return uncertainty_results, REPROJ_PX


def add_uncertainty_to_shapefile(output_shp, uncertainty_results, digitise_px, reproj_px,
                                 include_dem_error=False, dem_vertical_error_m=10.0,
                                 fov_sigma_deg=0.0):
    """Write the uncertainty summaries into the Cell 12 shapefile as extra columns."""
    # ── Write the numbers into the shapefile ─────────────────────────────────
    # Shapefile field names are limited to 10 characters, hence the abbreviations.
    print('\nAdding uncertainty attributes to the shapefile...')

    #using try/except so that if the write fails, cell computation and table still works
    try:
        # Read back Cell 12's shapefile and add the empty columns
        gdf = gpd.read_file(output_shp)                                                 #load shapefile from before
        new_cols = ['sig_cam_m', 'sig_dig_m', 'sig_fov_m', 'sig_rel_m', 'sig_rmin_m',   #name new columns
                    'sig_rmax_m', 'sig_grad', 'depth_vary', 'digi_px', 'fov_sig_d', 'reproj_px']
        if include_dem_error:
            new_cols += ['sig_dem_m', 'sig_abs_m', 'dem_verr_m']
        for col in new_cols:                                                            # create each column
            gdf[col] = None                                                             # fill with None for now

        # Fill each row from its matching summary. Features that were skipped above keep empty cells
        for i, row in gdf.iterrows():                                               # iterrows yields each row with its position i
            name = row['name']                                                      # the feature name stored in that row
            if name not in uncertainty_results:                                     # if no uncertainty was computed for it, leave alone and continue
                continue
            s = uncertainty_results[name]                                           # the summary value for this feature
            gdf.at[i, 'sig_cam_m']  = round(s['sigma_camera_mean'],   3)            # .at[row, column] writes a single cell. round() rounds the precision to specified amount.
            gdf.at[i, 'sig_dig_m']  = round(s['sigma_digitise_mean'], 3)
            gdf.at[i, 'sig_fov_m']  = round(s['sigma_fov_mean'],      3)
            gdf.at[i, 'sig_rel_m']  = round(s['sigma_relative_mean'], 3)
            gdf.at[i, 'sig_rmin_m'] = round(s['sigma_relative_min'],  3)
            gdf.at[i, 'sig_rmax_m'] = round(s['sigma_relative_max'],  3)
            gdf.at[i, 'sig_grad']   = round(s['sigma_grad'],          6)
            gdf.at[i, 'depth_vary'] = str(s['depth_varying'])
            gdf.at[i, 'digi_px']    = digitise_px
            gdf.at[i, 'fov_sig_d']  = round(float(fov_sigma_deg or 0.0), 3)
            gdf.at[i, 'reproj_px']  = round(reproj_px, 3)
            if include_dem_error:
                gdf.at[i, 'sig_dem_m']  = round(s['sigma_dem_mean'],      3)
                gdf.at[i, 'sig_abs_m']  = round(s['sigma_absolute_mean'], 3)
                gdf.at[i, 'dem_verr_m'] = dem_vertical_error_m

        # write to shapefile
        gdf.to_file(output_shp)
        print(f'  Updated: {output_shp}')
        print(f'  Added: {", ".join(new_cols)}')

    # otherwise if fail raise error
    except Exception as e:
        print(f'  Could not update the shapefile: {e}')
        print('  Run Cell 12 first to create it.')


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Orthorectification (optional)
# Cell 14 — Orthorectify the photo
# ══════════════════════════════════════════════════════════════════════════════

def load_stable_mask(stable_mask_path, dem):
    """
    Rasterise the stable terrain mask onto the output grid.

    Returns an (out_h, out_w) bool array (True = stable), or None if no mask is used.
    """
    out_h, out_w                = dem['out_h'], dem['out_w']
    out_transform, out_crs      = dem['out_transform'], dem['out_crs']

    # ── Load the stable terrain mask onto the output grid ─────────────────────────
    # The mask is defined in world space. Rasterising it onto the output grid (rather
    # than reprojecting anything else) means a simple array lookup answers "is this
    # output cell stable?" later on.
    stable_mask_raster = None                                               # (out_h, out_w) bool, or None if no mask is used

    if stable_mask_path is not None:                                        #if a mask was set
        print(f'  Loading stable terrain mask: {stable_mask_path}')
        ext = os.path.splitext(stable_mask_path)[1].lower()                 # split file name from extension, .lower() so TIF matches tif

        if ext in ('.tif', '.tiff', '.img'):                                ## if a raster mask was supplied...
            with rasterio.open(stable_mask_path) as msrc:                   # open it (then close automatically at end of block)
                if msrc.height == out_h and msrc.width == out_w:            # if already the same grid as the output
                    stable_mask_raster = msrc.read(1) != 0                  # read straight in. != 0 turns pixel values into True/False
                else:                                                       # otherwise not on same grid, resample to grid
                    mask_arr = np.zeros((out_h, out_w), dtype=np.uint8)         #destination array of correct shape
                    rasterio.warp.reproject(                                    #resample mask onto output grid
                        source=rasterio.band(msrc, 1), destination=mask_arr,    # from band 1 of file, onto new mask array
                        src_transform=msrc.transform, src_crs=msrc.crs,         # source cells location
                        dst_transform=out_transform, dst_crs=out_crs,           # destination cells location
                        resampling=rasterio.warp.Resampling.nearest)            # nearest neighbor resampling (stays binary)
                    stable_mask_raster = mask_arr != 0                          # covert to True/False
                    print('    Mask resampled onto the output grid')            #report this

        elif ext in ('.shp', '.gpkg', '.geojson', '.json'):                             ## if a vector mask was supplied...
            if not GEOPANDAS_OK:                                                        #check geopandas is installed first
                raise ImportError('geopandas is required to read a vector mask')
            from rasterio.features import rasterize                                     #package to convert to raster
            gdf_mask = gpd.read_file(stable_mask_path)                                  # read polygons into geopandas table
            if str(gdf_mask.crs) != str(out_crs):                                       # Reproject the polygons if they are not already in the DEM's CRS.
                gdf_mask = gdf_mask.to_crs(out_crs)
            stable_mask_raster = rasterize(                                             #convert to raster
                [(geom, 1) for geom in gdf_mask.geometry if geom is not None],          # each polygon gets value 1, everywhere else 0, then set to true/false
                out_shape=(out_h, out_w), transform=out_transform,
                fill=0, dtype=np.uint8) != 0
        else:                                                                       # otherwise format not available
            raise ValueError(f'Unsupported mask format: {ext}')

        #summary stats
        n_stable_out = stable_mask_raster.sum()                                     # counting stable cells
        print(f'    Stable cells   : {n_stable_out:,} '
              f'({n_stable_out/(out_h*out_w)*100:.1f}% of the output grid)')        #as a percent

    else:                                                                                   #otherwise no mask provided
        print('  No stable terrain mask — treating the whole scene as stable.')
        print('  (Set STABLE_MASK_PATH in Cell 2 to separate stable from unstable.)')

    return stable_mask_raster


def orthorectify(img_cv, n_bands, exclusion_mask, depth_image, rays_world, cam_world,
                 w_img, h_img, dem, stable_mask_raster):
    """
    Back-project every usable pixel to the ground and write its colour into the
    output grid (stable terrain only).

    Returns
    -------
    ortho_stack : (bands, out_h, out_w) uint8 orthorectified image
    ortho       : dict of the stable/unstable pixel mapping, needed by Cell 15
    """
    out_h, out_w  = dem['out_h'], dem['out_w']
    out_transform = dem['out_transform']

    # ── Back-project every usable pixel to world coordinates ──────────────────────
    print('\nBack-projecting pixels to world coordinates...')

    # Choose which pixels to orthorectify, ie.,
    # Pixels that are not masked AND hit the terrain.
    v_idx, u_idx = np.where(~exclusion_mask & (depth_image > 0))    # np.where on a 2D condition returns the row numbers and column numbers of every True position, as two matching lists.
    ray_idx  = v_idx * w_img + u_idx                                # flat index into the Cell 9 ray array
    ray_dirs = rays_world[ray_idx]                                  # unit direction per pixel
    dist_vals   = depth_image[v_idx, u_idx]                         # distance along that ray

    # Find the world XYZ of each hit
    # Walk along each ray from the camera by its own distance.
    xyz_world_all = cam_world + ray_dirs * dist_vals[:, np.newaxis] # [:, np.newaxis] makes the distances a column so each one scales all three components of its own direction vector.

    # Keep the source pixel coordinates — Cell 15 needs them as control points.
    u_all, v_all = u_idx, v_idx

    # ── Convert world coordinates to output raster cells ────────────────────────────────
    # Subtract the grid origin, divide by the cell size, round to the nearest whole cell: ie.,
    # Invert the affine transform. transform.c/.f are the grid origin, .a/.e the
    # pixel size (.e is negative for a north-up raster).
    col_out = (xyz_world_all[:, 0] - out_transform.c) / out_transform.a
    row_out = (xyz_world_all[:, 1] - out_transform.f) / out_transform.e
    col_int = np.round(col_out).astype(np.int64)                                 #round to the nearest whole cell, as an integer
    row_int = np.round(row_out).astype(np.int64)                                 #round to the nearest whole cell, as an integer

    # Discard anything landing outside the output grid.
    in_bounds = (col_int >= 0) & (col_int < out_w) & (row_int >= 0) & (row_int < out_h)         # inside the output raster
    valid_all = in_bounds & (u_all >= 0) & (u_all < w_img) & (v_all >= 0) & (v_all < h_img)     # and inside the photo

    # ── Split raster into stable and unstable ────────────────────────────────────────────
    if stable_mask_raster is not None:                                 # if a mask was loaded above, look up each landing cell in the mask
        col_safe = np.clip(col_int[valid_all], 0, out_w - 1)           # clip() forces values into a range, so the lookup is safe
        row_safe = np.clip(row_int[valid_all], 0, out_h - 1)           # same for rows
        stable_sel   = stable_mask_raster[row_safe, col_safe]          # True where that cell is inside the stable mask
        unstable_sel = ~stable_sel                                     # the rest
    else:                                                              # otherwise no mask, then everything counts as stable
        stable_sel   = np.ones(valid_all.sum(), dtype=bool)            # an array of all True
        unstable_sel = np.zeros(valid_all.sum(), dtype=bool)           # an array of all False


    # Index back into the full sized arrays.
    valid_idx    = np.where(valid_all)[0]
    stable_idx   = valid_idx[stable_sel]
    unstable_idx = valid_idx[unstable_sel]

    #report
    print(f'  Stable pixels   : {stable_sel.sum():,}')
    print(f'  Unstable pixels : {unstable_sel.sum():,}  (left as nodata for Cell 15)')

    # ── Record the stable mapping for Cell 15 ─────────────────────────────────────
    # Cell 15 fits a warp from these pairs: "output cell (col, row) was sampled from
    # photo pixel (u, v)". They are the ground truth it extrapolates from.
    stable_out_cols = col_int[stable_idx]      # output column of each stable pixel
    stable_out_rows = row_int[stable_idx]      # output row
    stable_img_u    = u_all[stable_idx]        # the photo column it came from
    stable_img_v    = v_all[stable_idx]        # the photo row it came from

    # ── Write the photo colours into the output grid ──────────────────────────────
    print(f'\nBuilding the orthorectified image ({out_w} x {out_h} px)...')

    if n_bands == 1:                                                                            #if greyscale
        img_gray = img_cv if img_cv.ndim == 2 else cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)     # already grey, or convert
        band1 = np.zeros((out_h, out_w), dtype=np.uint8)                                        # an empty output band. uint8 = 0-255
        band1[stable_out_rows, stable_out_cols] = img_gray[stable_img_v, stable_img_u]          # writes each pixel into own output cell
        ortho_stack = band1[np.newaxis, ...]                                                    # add a leading axis: (H, W) becomes (1, H, W), the layout rasterio wants
    else:                                                                                       #otherwise colour photo
        # img_cv is BGR (OpenCV order) but the GeoTIFF wants RGB, hence the 2,1,0.
        ortho_r = np.zeros((out_h, out_w), dtype=np.uint8)                                      # empty red band
        ortho_g = np.zeros((out_h, out_w), dtype=np.uint8)                                      # empty green band
        ortho_b = np.zeros((out_h, out_w), dtype=np.uint8)                                      # empty blue band
        ortho_r[stable_out_rows, stable_out_cols] = img_cv[stable_img_v, stable_img_u, 2]       # channel 2 of BGR is red
        ortho_g[stable_out_rows, stable_out_cols] = img_cv[stable_img_v, stable_img_u, 1]       # channel 1 is green
        ortho_b[stable_out_rows, stable_out_cols] = img_cv[stable_img_v, stable_img_u, 0]       # channel 0 is blue
        ortho_stack = np.stack([ortho_r, ortho_g, ortho_b], axis=0)                             # stack into (3, H, W)

    # Share of the output grid that received a pixel (several pixels can land on one cell, so count unique cells)
    coverage_pct = len(np.unique(stable_out_rows * out_w + stable_out_cols)) / (out_h * out_w) * 100

    ortho = {'stable_mask_raster': stable_mask_raster,
             'col_int': col_int, 'row_int': row_int,
             'stable_idx': stable_idx, 'unstable_idx': unstable_idx,
             'stable_sel': stable_sel, 'unstable_sel': unstable_sel,
             'stable_out_cols': stable_out_cols, 'stable_out_rows': stable_out_rows,
             'stable_img_u': stable_img_u, 'stable_img_v': stable_img_v,
             'coverage_pct': coverage_pct}
    return ortho_stack, ortho


def write_ortho_geotiff(output_path, ortho_stack, ortho, dem, nodata_value,
                        cam_world, fov, focal_px, err, image_path, stable_mask_path,
                        inlier_mask=None):
    """Write the orthorectified image as a GeoTIFF, tagged with the camera solution."""
    out_h, out_w           = dem['out_h'], dem['out_w']
    out_transform, out_crs = dem['out_transform'], dem['out_crs']
    cam_x, cam_y, cam_z    = cam_world
    err_in = err[inlier_mask] if inlier_mask is not None else err            # inlier GCPs (the ones the pose used)

    n_out_bands  = ortho_stack.shape[0]                                                         # 1 for greyscale, 3 for colour, used for writing below
    photo_interp = 'MINISBLACK' if n_out_bands == 1 else 'RGB'                                  # how GIS should interpret the bands


    # ── Write the GeoTIFF ─────────────────────────────────────────────────────────
    # The tags record the camera solution alongside the pixels, so the file can be
    # traced back to the model that made it.
    print(f'Writing -> {output_path}')                                              #annouce
    with rasterio.open(output_path, 'w',                                            #'w' opens the file for writing
                       driver='GTiff', height=out_h, width=out_w,                   # GeoTIFF format, and the grid size
                       count=n_out_bands, dtype=np.uint8,                           # how many bands, and 0-255 values
                       crs=out_crs, transform=out_transform,                        # the georeferencing: projection and grid position
                       compress='lzw', nodata=nodata_value,                         # lossless compression, and the "empty" value
                       photometric=photo_interp) as dst:                            # greyscale or RGB
        dst.write(ortho_stack)                                                      #  write all the bands
        dst.update_tags(                                                            # free-form metadata stored inside the file
            CAM_E=f'{cam_x:.3f}', CAM_N=f'{cam_y:.3f}', CAM_Z=f'{cam_z:.3f}',       # where the camera was
            FOV_DEG=f'{fov:.4f}', FOCAL_PX=f'{focal_px:.2f}',                       # the optics used
            REPROJ_ERR=f'{err_in.mean():.3f}px',                                    # how well the model fitted the GCPs (inliers)
            REPROJ_ERR_ALL=f'{err.mean():.3f}px',                                   # the same over all GCPs, incl. RANSAC outliers
            N_GCP_INLIERS=f'{len(err_in)}/{len(err)}',                              # how many GCPs the pose was solved from
            SRC_IMAGE=os.path.basename(image_path),                                 # basename = the file name without its folders
            METHOD='ray_mesh_intersection',                                         # how these pixels were placed
            STABLE_MASK=str(stable_mask_path))                                      # which mask was applied, if any

    size_mb = os.path.getsize(output_path) / 1e6                                    # file size in megabytes

    #summary
    print(f'\n{"-"*52}')
    print(f'  Output   : {os.path.abspath(output_path)}  ({size_mb:.1f} MB)')
    print(f'  Stable   : {ortho["stable_sel"].sum():,} pixels written')
    print(f'  Unstable : {ortho["unstable_sel"].sum():,} pixels left as nodata')
    print(f'{"-"*52}')


def plot_ortho(img_rgb, ortho_stack, coverage_pct):
    """Preview the orthorectified image beside the photo."""
    n_out_bands = ortho_stack.shape[0]

    # ── Preview orthorectification here ───────────────────────────────────────────────────────────────────
    preview = np.moveaxis(ortho_stack, 0, -1)                               # (bands, H, W) -> (H, W, bands)
    if n_out_bands == 1:                                                    # matplotlib wants a plain 2D array for greyscale
        preview = preview[:, :, 0]                                          # drop the length-1 band axis
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))                         # one row of two panels
    axes[0].imshow(img_rgb)                                                 # left: the original photo
    axes[0].set_title('Original photo')
    axes[0].axis('off')
    axes[1].imshow(preview, cmap='grey' if n_out_bands == 1 else None)      # right: the ortho. cmap only matters for greyscale
    axes[1].set_title(f'Orthorectified ({coverage_pct:.1f}% coverage)\n'
                      f'nodata = sky, hidden ground, and unstable terrain')
    axes[1].axis('off')                               # no axis numbers
    plt.tight_layout(); plt.show()                    # tidy spacing, then display


# ══════════════════════════════════════════════════════════════════════════════
# Cell 15 — Fill unstable terrain by warping
# ══════════════════════════════════════════════════════════════════════════════

# Build a design matrix: one column per term of the polynomial, one row
# per point. Fitting is then a least-squares solve for the coefficients,
# and predicting is a matrix multiply.
def poly_design(xy, order):                                                     # a helper used only by the polynomial warp
    """Design matrix of all monomials x^i * y^j with i + j <= order."""
    x, y = xy[:, 0], xy[:, 1]                                                   # split the two columns into separate names
    cols_dm = [np.ones(len(x))]                                                 # the constant term: a column of 1s
    for deg in range(1, order + 1):                                             # each total degree, 1 up to the order
        for k in range(deg + 1):                                                # each split of that degree between x and y
            cols_dm.append((x ** (deg - k)) * (y ** k))
    return np.stack(cols_dm, axis=1)                                            # glue the columns into a matrix


# Function to blend the four neighbours: across the top pair, across the bottom pair, then between those two results.
# Returns a flat uint8 array of one brightness value per cell being filled,
# in the same order as pred_u/pred_v — every cell at once, not one at a time.
def bilinear_sample(ch, u0, v0, u1, v1, du, dv):
    """Bilinear interpolation of one channel at (pred_u, pred_v)."""
    top    = ch[v0, u0] * (1 - du) + ch[v0, u1] * du                        # blend the two pixels along the top edge
    bottom = ch[v1, u0] * (1 - du) + ch[v1, u1] * du                        # blend the two along the bottom edge
    return np.clip(top * (1 - dv) + bottom * dv, 0, 255).astype(np.uint8)   # blend those two, then force into 0-255


def warp_fill(ortho, output_path, img_cv, w_img, h_img, dem, nodata_value,
              warp_method='tps', warp_poly_order=2, warp_max_control_px=5000,
              output_warp_path=None):
    """
    Fill the unstable terrain left as nodata by Cell 14, by warping the photo
    from the stable control points.

    ortho : the dict returned by orthorectify (None if Cell 14 has not been run)

    Returns a dict of results for plot_warp_fill, or None if there was nothing to fill.
    """
    out_h, out_w           = dem['out_h'], dem['out_w']
    out_transform, out_crs = dem['out_transform'], dem['out_crs']

    # ──────────────────────────────────────────────────────────────────────────────
    # Build the output path, preserving whatever extension OUTPUT_PATH used.
    _base, _ext   = os.path.splitext(output_path)
    warp_out_path = output_warp_path if output_warp_path else f'{_base}_warp{_ext}'

    # ── Precondition checks ─────────────────────────────────────────────────────────────
    if ortho is None:                                                                   # Cell 14 never ran
        print('Cell 14 has not been run — run it first, with STABLE_MASK_PATH set.')
        return None
    if ortho['stable_mask_raster'] is None:                                             # Cell 14 ran, but without a mask
        print('No stable terrain mask was used in Cell 14, so every pixel was already')
        print('written and there is nothing to fill. Set STABLE_MASK_PATH in Cell 2')
        print('and re-run Cells 14-15 if you need this step.')
        return None
    if ortho['unstable_sel'].sum() == 0:                                                # a mask was used but flagged nothing
        print('The mask marked no unstable pixels — nothing to fill.')
        print('Check that STABLE_MASK_PATH really outlines the changed area.')
        return None

    # Stable/unstable mapping from Cell 14
    col_int, row_int                 = ortho['col_int'], ortho['row_int']
    unstable_idx                     = ortho['unstable_idx']
    stable_sel, unstable_sel         = ortho['stable_sel'], ortho['unstable_sel']
    stable_out_cols, stable_out_rows = ortho['stable_out_cols'], ortho['stable_out_rows']
    stable_img_u, stable_img_v       = ortho['stable_img_u'], ortho['stable_img_v']

    # otherwise all good, print initial report and continue
    print('Cell 15: warp fill for unstable terrain')
    print(f'  Method          : {warp_method}')
    if warp_method == 'polynomial':
        print(f'  Polynomial order: {warp_poly_order}')
    print(f'  Control points  : {len(stable_out_cols):,} stable pixels available')
    print(f'  To fill         : {unstable_sel.sum():,} unstable pixels')
    print(f'  Output          : {warp_out_path}\n')

    # ── Start from the Cell 14 orthorectified image ──────────────────────────────────────────
    # add to orthorectified image rather than rebuilding it.
    with rasterio.open(output_path) as src:             # reopen what Cell 14 wrote
        ortho_fill   = src.read()                       # format (bands, H, W)
        n_fill_bands = src.count                        # 1 for greyscale, 3 for colour (needed for loop further down)

    # ── Assemble the control points ───────────────────────────────────────────
    # Each control point pairs an output cell with the photo pixel it came from.
    ctrl_raster = np.stack([stable_out_cols.astype(np.float64),                             # list of control points from stable columns and rows from Cell 14
                            stable_out_rows.astype(np.float64)], axis=1)  # (N, 2)
    ctrl_img_u  = stable_img_u.astype(np.float64)                                           # collect control image coord cols
    ctrl_img_v  = stable_img_v.astype(np.float64)                                           # collect control image coord rows

    # ── Thin the control points if there are too many ─────────────────────────
    # Sampling at random would over-represent whichever area has the most stable
    # pixels. Instead, lay a grid over the extent and take a few points from each
    # occupied cell, so the points span the stable terrain evenly. Clustered
    # control points also make the TPS matrix near-singular.

    # Work out a roughly square grid of bins, label every control point with
    # the bin it falls in, then draw a fixed quota at random from each bin. Bins
    # that were empty leave the quota short, so a random top-up follows.
    if len(ctrl_raster) > warp_max_control_px:                                                          # only thin when over the cap
        n_grid   = int(np.ceil(np.sqrt(warp_max_control_px)))                                           # number of grid cells is an integer of the square root of the max control points, rounded up using ceil
        col_bins = np.linspace(ctrl_raster[:, 0].min(), ctrl_raster[:, 0].max(), n_grid + 1)            # linspace gives n_grid+1 evenly spaced edges, which is n_grid bins.
        row_bins = np.linspace(ctrl_raster[:, 1].min(), ctrl_raster[:, 1].max(), n_grid + 1)
        col_cell = np.clip(np.digitize(ctrl_raster[:, 0], col_bins) - 1, 0, n_grid - 1)                 # digitize says which bin each value falls in (counting from 1, hence -1).
        row_cell = np.clip(np.digitize(ctrl_raster[:, 1], row_bins) - 1, 0, n_grid - 1)
        cell_id  = row_cell * n_grid + col_cell                                                         # one number identifying each bin

        # Draw the quota from each occupied bin
        rng          = np.random.default_rng(32)                                                        # random number generator. fixed seed for reproducible results
        pts_per_cell = max(1, warp_max_control_px // (n_grid * n_grid))                                 # // divides and rounds down; at least 1
        sel_idx      = []                                                                               # will collect the chosen positions
        for cid in np.unique(cell_id):                                                                  # loop over the bins that contain points
            in_cell = np.where(cell_id == cid)[0]                                                       # positions of the points in this bin
            sel_idx.append(rng.choice(in_cell, min(pts_per_cell, len(in_cell)),                         # take the quota, or all of them
                                      replace=False))                                                   # replace=False means no duplicates
        sel_idx = np.concatenate(sel_idx)                                                               # join the per-bin picks into one list


        # Empty grid cells leave the quota unfilled — top up at random.
        if len(sel_idx) < warp_max_control_px:                                                          # if short of the target
            remaining = np.setdiff1d(np.arange(len(ctrl_raster)), sel_idx)                              # every position not already chosen
            extra = rng.choice(remaining,                                                               # pick from those...
                               min(warp_max_control_px - len(sel_idx), len(remaining)),                 # ...just enough to reach the cap
                               replace=False)
            sel_idx = np.concatenate([sel_idx, extra])                                                  # add them to the selection

        # Apply the same selection to all three arrays so the pairs stay matched
        ctrl_raster = ctrl_raster[sel_idx]               # keep only the chosen control points
        ctrl_img_u  = ctrl_img_u[sel_idx]                # and their photo columns
        ctrl_img_v  = ctrl_img_v[sel_idx]                # and their photo rows

        #report
        print(f'  Thinned to {len(ctrl_raster):,} control points on a '
              f'{n_grid}x{n_grid} grid (from {len(stable_out_cols):,})')

    #otherwise, it was already under the limit and no thinning needed
    else:
        print(f'  Using all {len(ctrl_raster):,} control points')

    # ── Query points: the unstable output cells ───────────────────────────────
    # list the cells of the raster that need to be orthorectified, same format as above
    query_raster = np.stack([col_int[unstable_idx].astype(np.float64),
                             row_int[unstable_idx].astype(np.float64)], axis=1)
    n_query = len(query_raster)   #number of cells to be warped

    # ── Fit the warp and predict where to sample ──────────────────────────────
    # Use control points to determine raster cell > photo pixel relationship, then apply to 'unstable' pixels
    # Two options: TPS bends freely to pass through every control
    # point. Polynomial is rigid and cannot follow local detail, but never
    # over-extrapolates. Results in both cases are pred_u and pred_v:
    # one predicted photo column and row per cell to be filled
    print(f'  Fitting {warp_method.upper()} on {len(ctrl_raster):,} control points...')


    # if thin plate spline selected...
    if warp_method == 'tps':
        # smoothing=0 makes the spline pass exactly through the control points,
        # which is what we want. It can fail on near-collinear or clustered
        # points, so fall back to progressively more regularisation.

        # The loop tries some smoothing values one at a time. ie. fit both splines, and if
        # the matrix cannot be solved, move on to the next value.
        for smoothing in (0, 1e-6, 1e-3):
            try:
                with warnings.catch_warnings():                                             # ignore warnings if fail and move on to next
                    warnings.simplefilter('ignore')
                    tps_u = RBFInterpolator(ctrl_raster, ctrl_img_u,                        # fit output cell > photo column
                                            kernel='thin_plate_spline',
                                            smoothing=smoothing)
                    tps_v = RBFInterpolator(ctrl_raster, ctrl_img_v,                        # fit output cell > photo row
                                            kernel='thin_plate_spline',
                                            smoothing=smoothing)
                if smoothing > 0:                                                           #if smoothing was needed, report this
                    print(f'  smoothing=0 was singular — using smoothing={smoothing} '
                          f'(still near-exact)')
                break                                                                       # success, leave the loop
            except np.linalg.LinAlgError:                                                   # fail, could not solve
                continue                                                                    # try the next smoothing value
        else:                                                                               # otherwise every smoothing value failed and the loop didnt successfully 'break'
            raise RuntimeError(                                                                     # so raise an error
                'TPS fitting failed even with regularisation. Reduce '
                'WARP_MAX_CONTROL_PX, or switch to WARP_METHOD="polynomial".')

        # Now that we have a solution, apply the two fitted splines to the cells that need filling.
        pred_u = tps_u(query_raster)
        pred_v = tps_v(query_raster)


    #otherwise, if polynomial selected...
    elif warp_method == 'polynomial':

        A_ctrl  = poly_design(ctrl_raster,  warp_poly_order)                            # the terms evaluated at the control points
        A_query = poly_design(query_raster, warp_poly_order)                            # and at the cells to be filled
        with warnings.catch_warnings():                                                 # ignore any conditioning warnings
            warnings.simplefilter('ignore')

            # lstsq finds the coefficients that best fit the data. *_ absorbs the
            # extra diagnostic values it returns alongside them.
            coeff_u, *_ = np.linalg.lstsq(A_ctrl, ctrl_img_u, rcond=None)   # coefficients for the column fit
            coeff_v, *_ = np.linalg.lstsq(A_ctrl, ctrl_img_v, rcond=None)   # and for the row fit

        # after polynomal coefficients solved, recover the predicted image pixels
        pred_u = A_query @ coeff_u                    # @ multiplies the matrix by the coefficients
        pred_v = A_query @ coeff_v                    # giving one predicted row per cell

        print(f'  Polynomial order {warp_poly_order} ({A_ctrl.shape[1]} terms)')   # shape[1] = number of columns = terms

    #otherwise, warp setting didnt work?
    else:
        raise ValueError(f'WARP_METHOD must be "tps" or "polynomial", got "{warp_method}"')



    # ── Check the predictions land inside the photo ───────────────────────────
    # A warp asked to extrapolate far beyond its control points can
    # predict pixels that do not exist. Worth checking.
    # Count out-of-bounds predictions before clipping and a lot of them means the
    # warp is extrapolating badly.
    n_oob = ((pred_u < 0) | (pred_u >= w_img) |             #number of out of bounds columns/rows
             (pred_v < 0) | (pred_v >= h_img)).sum()
    pred_u = np.clip(pred_u, 0, w_img - 1.001)              #force columns and rows back inside the frame, and leave .001 reoom to interpolate
    pred_v = np.clip(pred_v, 0, h_img - 1.001)
    if n_oob > 0:                                           #if some were clipped, report.
        print(f'  {n_oob:,} predictions fell outside the photo and were clipped')
        print('    (a few is normal near the frame edge; many means the warp is')
        print('     extrapolating too far from the stable terrain)')

    # Confirm that TPS method can reproduce its own control points by measureing residual
    # For TPS with smoothing=0 the residual at the control points should be ~0.
    # Anything else means the fit fell back to regularisation.
    if warp_method == 'tps':
        resid = np.sqrt((tps_u(ctrl_raster) - ctrl_img_u) ** 2 +
                        (tps_v(ctrl_raster) - ctrl_img_v) ** 2)
        print(f'  Residual at control points: mean={resid.mean():.3f} px  '
              f'max={resid.max():.3f} px  (should be ~0)')



    # ── Sample the photo at the predicted positions ───────────────────────────
    # Predictions are fractional, so interpolate between the four surrounding
    # pixels rather than rounding.
    # u0/v0 are the whole-pixel corner up and left of the prediction, u1/v1 the
    # one down and right; du/dv are how far between them the prediction sits.
    print(f'\n  Sampling {n_query:,} pixels from the photo...')
    u0 = np.clip(np.floor(pred_u).astype(np.int32), 0, w_img - 1)               #floor rounds down
    v0 = np.clip(np.floor(pred_v).astype(np.int32), 0, h_img - 1)
    u1 = np.clip(u0 + 1, 0, w_img - 1)
    v1 = np.clip(v0 + 1, 0, h_img - 1)
    du = (pred_u - np.floor(pred_u)).astype(np.float32)                         # sub-pixel offsets
    dv = (pred_v - np.floor(pred_v)).astype(np.float32)

    # --- Write each sampled colour into the cell it belongs to -------------
    fill_cols = col_int[unstable_idx]               #output col
    fill_rows = row_int[unstable_idx]               #output row

    ortho_warp = ortho_fill.copy()                  #start from a copy of cell 14 output to leave untouched

    if n_fill_bands == 1:                                                                           #if greyscale...
        img_gray = img_cv if img_cv.ndim == 2 else cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)         # already grey or convert
        ortho_warp[0][fill_rows, fill_cols] = bilinear_sample(img_gray, u0, v0, u1, v1, du, dv)     # then fill the sinle band
    else:                                                                                           # if colour...
        ortho_warp[0][fill_rows, fill_cols] = bilinear_sample(img_cv[:, :, 2], u0, v0, u1, v1, du, dv)  # R      # One call to bilinear_sample per band; img_cv is BGR, so the channels aretaken in reverse to write R, G, B.
        ortho_warp[1][fill_rows, fill_cols] = bilinear_sample(img_cv[:, :, 1], u0, v0, u1, v1, du, dv)  # G
        ortho_warp[2][fill_rows, fill_cols] = bilinear_sample(img_cv[:, :, 0], u0, v0, u1, v1, du, dv)  # B

    # measure the coverage (accounting for multiple pixels landing on same output cell)
    unique_stable   = len(np.unique(stable_out_rows * out_w + stable_out_cols))
    unique_unstable = len(np.unique(fill_rows * out_w + fill_cols))
    coverage_pct_warp = (unique_stable + unique_unstable) / (out_h * out_w) * 100

    # ── Write/save the filled ortho ────────────────────────────────────────────────

    #first set how GIS should read the bands
    photo_interp = 'MINISBLACK' if n_fill_bands == 1 else 'RGB'

    #write the file
    with rasterio.open(warp_out_path, 'w',                             # open new file and export with these settings
                       driver='GTiff', height=out_h, width=out_w,      # GeoTIFF, same grid size as before
                       count=n_fill_bands, dtype=np.uint8,             # same band count, 0-255 values
                       crs=out_crs, transform=out_transform,           # the same georeferencing
                       compress='lzw', nodata=nodata_value,            # lossless compression and the empty value
                       photometric=photo_interp) as dst:
        dst.write(ortho_warp)                                     # write the pixels
        dst.update_tags(                                          # record how the fill was done
            WARP_METHOD=warp_method,                              # 'tps' or 'polynomial'
            WARP_CTRL_PTS=len(ctrl_raster),                       # how many control points were used
            WARP_POLY_ORDER=str(warp_poly_order) if warp_method == 'polynomial' else 'N/A',   # only meaningful for one method
            STABLE_SOURCE=output_path)                            # which file this was built from

    size_mb = os.path.getsize(warp_out_path) / 1e6                # file size in megabytes

    #report outputs
    print(f'\n{"-"*52}')
    print(f'  Output   : {os.path.abspath(warp_out_path)}  ({size_mb:.1f} MB)')
    print(f'  Stable   : {stable_sel.sum():,} pixels (ray-traced)')
    print(f'  Filled   : {n_query:,} pixels (warped, approximate)')
    print(f'  Coverage : {coverage_pct_warp:.1f}%')
    print(f'{"-"*52}')
    print('\nFilled ortho written. Part 3 complete.')

    return {'ortho_fill': ortho_fill, 'ortho_warp': ortho_warp, 'n_fill_bands': n_fill_bands,
            'fill_rows': fill_rows, 'fill_cols': fill_cols,
            'stable_out_rows': stable_out_rows, 'stable_out_cols': stable_out_cols,
            'coverage_pct_warp': coverage_pct_warp, 'n_query': n_query,
            'out_h': out_h, 'out_w': out_w, 'warp_out_path': warp_out_path}


def plot_warp_fill(warp):
    """Before / after / what-was-filled panels for the warp fill (warp = warp_fill result)."""
    ortho_fill, ortho_warp = warp['ortho_fill'], warp['ortho_warp']
    n_fill_bands           = warp['n_fill_bands']
    fill_rows, fill_cols   = warp['fill_rows'], warp['fill_cols']
    stable_out_rows        = warp['stable_out_rows']
    stable_out_cols        = warp['stable_out_cols']
    coverage_pct_warp      = warp['coverage_pct_warp']
    n_query                = warp['n_query']
    out_h, out_w           = warp['out_h'], warp['out_w']

    # ── VISUAL CHECK: Before / after / what-was-filled ──────────────────────────────────────
    #Panels 1 and 2 show the hole and the fill, panel 3 marks exactly which pixels are interpolated
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))               # one row of three panels

    disp_stable = np.moveaxis(ortho_fill, 0, -1)                  # (bands, H, W) -> (H, W, bands) for display # rasterio stores bands first, matplotlib wants them last, hence moveaxis.
    if n_fill_bands == 1:                                         # matplotlib wants plain 2D for greyscale
        disp_stable = disp_stable[:, :, 0]                        # drop the length-1 band axis
    axes[0].imshow(disp_stable, cmap='grey' if n_fill_bands == 1 else None)   # left panel: before
    axes[0].set_title('Cell 14: stable terrain only')             # its title
    axes[0].axis('off')                                           # no axis numbers

    disp_warp = np.moveaxis(ortho_warp, 0, -1)                    # the same rearrangement for the filled version
    if n_fill_bands == 1:                                                     # matplotlib wants a plain 2D array for greyscale
        disp_warp = disp_warp[:, :, 0]                            # drop the band axis for greyscale
    axes[1].imshow(disp_warp, cmap='grey' if n_fill_bands == 1 else None)   # middle panel: after
    axes[1].set_title(f'Cell 15: filled\n({coverage_pct_warp:.1f}% coverage, '   # title line 1
                      f'{n_query:,} pixels added)')                              # title line 2
    axes[1].axis('off')                                           # no axis numbers

    fill_map = np.zeros((out_h, out_w), dtype=np.uint8)                       # 0 = nodata
    fill_map[stable_out_rows, stable_out_cols] = 1                            # 1 = ray-traced (stable)
    fill_map[fill_rows, fill_cols] = 2                                        # 2 = warped (interpolated)
    fill_cmap = mcolors.ListedColormap(['white', 'steelblue', 'orange'])      # one colour per class
    axes[2].imshow(fill_map, cmap=fill_cmap, vmin=0, vmax=2, interpolation='nearest')   # right panel: what was filled
    axes[2].set_title('What was filled\nblue = ray-traced, orange = warped (interpolated)')
    axes[2].axis('off')                                           # no axis numbers
    plt.tight_layout()
    plt.show()
