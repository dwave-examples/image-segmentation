import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

# Define our weight function
def weight(a, b, img):
    _, cols, _ = img.shape
    diff = img[int(a/cols), a%cols, :] - img[int(b/cols), b%cols, :]
    return np.sum(np.square(diff))

# Convert single index into tuple indices
def unindexing(a):
    rows, cols, _ = img.shape
    y1 = a % cols
    x1 = int(a/cols)
    return (x1, y1)

if len(sys.argv) > 1:
    # Read in image file specified
    data_file_name = sys.argv[1]
    random = False

    print("\nReading in your image...")
    img = cv2.imread(data_file_name)

    response_2 = input("\n\tEnter number of segments > ")
    try:
        num_segments = int(response_2)
    except ValueError:
        print("Must input an integer.")
        num_segments = int(input("\n\tEnter number of segments > "))
else:
    # Generate a random image with segments
    print("\nCreating random image...")
    random = True

    # Collect user input on size of problem
    response_1 = input("\n\tEnter image dimensions > ")
    try:
        dims = int(response_1)
    except ValueError:
        print("Must input an integer.")
        dims = int(input("\n\tEnter image dimensions > "))

    response_2 = input("\n\tEnter number of segments > ")
    try:
        num_segments = int(response_2)
    except ValueError:
        print("Must input an integer.")
        num_segments = int(input("\n\tEnter number of segments > "))

    img = np.zeros((dims,dims,3), np.uint8)
    img_rows = np.sort(np.random.choice(dims, num_segments, replace=False))
    img_cols = np.sort(np.random.choice(dims, num_segments, replace=False))
    for num in range(num_segments-1):
        color = np.random.randint(0,255,3)
        img[img_rows[num]:, img_cols[num]:, :] = color

# Build the DQM and set biases according to pixel similarity
print("\nPreparing DQM object...")
rows, cols, _ = img.shape
linear_biases = np.zeros(rows*cols*num_segments)
case_starts = np.arange(rows*cols)*num_segments
num_interactions = rows*cols*(rows*cols-1)*num_segments/2
qb_rows = []
qb_cols = []
qb_biases = []
for i in range(rows*cols):
    for j in range(i+1, rows*cols):
        for case in range(num_segments):
            qb_rows.append(i*num_segments+case)
            qb_cols.append(j*num_segments+case)
            qb_biases.append(weight(i,j,img))
quadratic_biases = (np.asarray(qb_rows), np.asarray(qb_cols), np.asarray(qb_biases))
dqm = DiscreteQuadraticModel.from_numpy_vectors(case_starts, linear_biases, quadratic_biases)

# Initialize the DQM solver
print("\nRunning DQM solver...")
sampler = LeapHybridDQMSampler()

# Solve the problem using the DQM solver
sampleset = sampler.sample_dqm(dqm)

# Get the first solution
sample = sampleset.first.sample

print("\nProcessing solution...")
im_segmented = np.zeros((rows,cols))
for key, val in sample.items():
    x, y = unindexing(key)
    im_segmented[x,y] = val

if random:
    row_indices = [1+i for i in range(rows-1)]
    row_indices.append(0)
    im_segmented_rowwrap = im_segmented[row_indices,:]

    col_indices = [1+i for i in range(cols-1)]
    col_indices.append(0)
    im_segmented_colwrap = im_segmented[:,col_indices]

    im_seg_rowdiff = im_segmented - im_segmented_rowwrap
    im_seg_coldiff = im_segmented - im_segmented_colwrap
    segmented_image = np.ones((rows, cols, 3), np.uint8)*255
    segmented_image[im_seg_rowdiff != 0] = (255, 0, 0)
    segmented_image[im_seg_coldiff != 0] = (255, 0, 0)
else:
    segmented_image = im_segmented

fig, (ax1, ax2) = plt.subplots(1, 2)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax1.imshow(img)
ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)
ax2.imshow(segmented_image, cmap='Greys')
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)
plt.savefig("output.png")
plt.show()

