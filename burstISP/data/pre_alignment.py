# Modified from homography_alignment.py from FBANet
import cv2
import os
import numpy as np
from functools import partial

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

LR_trainpatch_path = '/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_trainpatch'
trainpatch_save_path = '/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_trainpatch_aligned'
LR_testpatch_path = '/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch'
testpatch_save_path = '/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch_aligned'

def process_one_frame(i, im1, LR_patch_path, LR_list, LR_number1, LR_number2, save_LR_path):
    # im2_path = '{}/{}/{}_MFSR_Sony_{:04d}_x4_{:02d}.png'.format(LR_patch_path, LR_list, LR_number1, LR_number2, i)
    im2_path = '{}/{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(LR_patch_path, LR_list, LR_number1, LR_number2, i)
    im2 = cv2.imread(im2_path)

    if not os.path.exists(im2_path):
        logs = open('DRealBSR_test.txt', 'a')
        logs.write(im2_path)
        logs.write('\n')
        logs.close()
        return

    print("processing image {}".format(im2_path))

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    # warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(save_LR_path, LR_number1, LR_number2, i),
                    im2_aligned)

    except:
        cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(save_LR_path, LR_number1, LR_number2, i), im2)
        print("An error occured when ECC not converge")

def process_one_image(LR_list, LR_patch_path, save_path, save_gt_path):
    # Removed the global declarations
    LR_number1 = LR_list.split('_')[0]
    LR_number2 = int(LR_list.split('_')[-1])
    base_frame_path = '{}/{}/{}_MFSR_Sony_{:04d}_x1_00.png'.format(LR_patch_path, LR_list, LR_number1, LR_number2)
    gt_frame_path = '{}/{}/{}_MFSR_Sony_{:04d}_x4_rgb.png'.format(LR_patch_path, LR_list, LR_number1, LR_number2)

    im1 = cv2.imread(base_frame_path)
    gt_img = cv2.imread(gt_frame_path)

    save_LR_path = os.path.join(save_path, LR_list)
    os.makedirs(save_LR_path, exist_ok=True)

    save_hr_path = os.path.join(save_gt_path, LR_list)
    os.makedirs(save_hr_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=16) as t: 
        for i in range(1, 14):
            t.submit(lambda cxp:process_one_frame(*cxp),(i, im1, LR_patch_path, LR_list, LR_number1, LR_number2, save_LR_path))

    cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(save_LR_path, LR_number1, LR_number2, 0), im1)
    cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x4.png'.format(save_hr_path, LR_number1, LR_number2, 0), gt_img)

def run_alignment_for_dataset(source_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    # Re-using the target_path for both LR and HR for simplicity, 
    
    # Create a partial function to lock in the paths for the pool map
    process_func = partial(process_one_image, LR_patch_path=source_path, save_path=target_path, save_gt_path=target_path)
    
    pool = multiprocessing.Pool(16)
    pool.map(process_func, os.listdir(source_path))
    pool.close()
    pool.join()

def main():
    print("Processing Training Patches...")
    run_alignment_for_dataset(LR_trainpatch_path, trainpatch_save_path)
    
    print("Processing Testing Patches...")
    run_alignment_for_dataset(LR_testpatch_path, testpatch_save_path)

if __name__ == '__main__':
    main()