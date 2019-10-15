import sys
sys.path.append('/home/luoqibin/Desktop/video-quality')
import numpy
import argparse
import re
import scipy.misc
import subprocess
import os.path
import vifp
import ssim
import psnr
import niqe
import reco
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
ref_path = '/Users/lche/Desktop/test_results'
# dist_path = '/data/test_results/cylce_Deblur/test_latest/images'
ref_file = ''
dist_file = ''

# ref = scipy.misc.imread(ref_file, flatten=True).astype(numpy.float32)

quality_values = []
# ref_values = []
# dist_values = []
# size_values = []
# vifp_values = []
# ssim_values = []
# psnr_values = []
# niqe_values = []
# reco_values = []
vifp_sum = 0
ssim1_sum = 0
psnr1_sum = 0
ssim2_sum = 0
psnr2_sum = 0
nige_sum = 0
reco_sum = 0
num = 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir')
    parser.add_argument('save_dir')
    with open('esstimate.txt', 'a') as f:
        # f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ('num', 'ref_file', 'ref_size', 'dist_file', 'dist_size', 'vifp', 'ssim', 'psnr', 'reco'))
        for root, dirs, files in os.walk(ref_path):
            for name in files:
                if '_5.png' in os.path.split(name)[1]:
                    sharp_file = os.path.join(root, name)
                    dist1_file = sharp_file.replace('_5', '_1')
                    dist2_file = sharp_file.replace('_5', '_2')
                    dist3_file = sharp_file.replace('_5', '_3')
                    dist4_file = sharp_file.replace('_5', '_4')
                    if os.path.isfile(sharp_file) and os.path.isfile(dist1_file):
                        print(dist1_file)
                        dist1 = scipy.misc.imread(dist1_file, flatten=True).astype(numpy.float32)
                        dist2 = scipy.misc.imread(dist2_file, flatten=True).astype(numpy.float32)
                        dist3 = scipy.misc.imread(dist3_file, flatten=True).astype(numpy.float32)
                        dist4 = scipy.misc.imread(dist4_file, flatten=True).astype(numpy.float32)
                        sharp = scipy.misc.imread(sharp_file, flatten=True).astype(numpy.float32)
                        # print(ref_file + ' ' + dist_file + ':')
                        num += 1
                        # vifp1 = vifp.vifp_mscale(ref, dist)
                        ssim1 = ssim.ssim_exact(sharp / 255, dist1 / 255)
                        ssim2 = ssim.ssim_exact(sharp / 255, dist2 / 255)
                        ssim3 = ssim.ssim_exact(sharp / 255, dist3 / 255)
                        ssim4 = ssim.ssim_exact(sharp / 255, dist4 / 255)
                        psnr1 = psnr.psnr(sharp, dist1)
                        psnr2 = psnr.psnr(sharp, dist2)
                        psnr3 = psnr.psnr(sharp, dist3)
                        psnr4 = psnr.psnr(sharp, dist4)
                        # reco1 = reco.reco(ref / 255, dist / 255)
                        # vifp_sum += vifp1
                        # reco_sum += reco1
                        # ref_values.append(ref_file)
                        # dist_values.append(dist_file)
                        # vifp_values.append(vifp1)
                        # ssim_values.append(ssim1)
                        # psnr_values.append(psnr1)
                        # reco_values.append(reco1)
                        f.write("%s:\n" % sharp_file)
                        f.write("%f\t%f\t%f\t%f\n" % (ssim1, ssim2, ssim3, ssim4))
                        f.write("%f\t%f\t%f\t%f\n" % (psnr1, psnr2, psnr3, psnr4))
                        # f.write("%d\t%s\t%d\t%s\t%d\t%f\t%f\t%f\t%f\n" % (num, ref_file, int(os.path.getsize(ref_file)/1024), dist_file, int(os.path.getsize(dist_file)/1024), vifp1, ssim1, psnr1, reco1))
        # print("average_ssim of deblur2sharp = %f" % (ssim1_sum / num))
        # print("average_ssim of blured2sharp = %f" % (ssim2_sum / num))
        # print("average_psnr of deblur2sharp = %f" % (psnr1_sum / num))
        # print("average_psnr of blured2sharp = %f" % (psnr2_sum / num))
        # f.write("average_ssim of deblured2sharp = %f\n" % (ssim1_sum / num))
        # f.write("average_ssim of blured2sharp = %f\n" % (ssim2_sum / num))
        # f.write("average_psnr of deblured2sharp = %f\n" % (psnr1_sum / num))
        # f.write("average_psnr of blured2sharp = %f\n" % (psnr2_sum / num))
        f.close()
    # print("average of vifp = %f", vifp_sum / num)
    # print("average of reco = %f", reco_sum / num)



# for quality in range(0, 101, 1):
#     subprocess.check_call('gm convert %s -quality %d %s'%(ref_file, quality, dist_file), shell=True)
#     file_size = os.path.getsize(dist_file)
#
#     dist = scipy.misc.imread(dist_file, flatten=True).astype(numpy.float32)
#
#     quality_values.append( quality )
#     size_values.append( int(file_size/1024) )
#     vifp_values.append( vifp.vifp_mscale(ref, dist) )
#     ssim_values.append( ssim.ssim_exact(ref/255, dist/255) )
#     psnr_values.append( psnr.psnr(ref, dist) )
#     # niqe_values.append( niqe.niqe(dist/255) )
#     reco_values.append( reco.reco(ref/255, dist/255) )

# plt.plot(quality_values, vifp_values, label='VIFP')
# plt.plot(quality_values, ssim_values, label='SSIM')
# # plt.plot(niqe_values, label='NIQE')
# plt.plot(quality_values, reco_values, label='RECO')
# plt.plot(quality_values, numpy.asarray(psnr_values)/100.0, label='PSNR/100')
# plt.legend(loc='lower right')
# plt.xlabel('JPEG Quality')
# plt.ylabel('Metric')
# plt.savefig('jpg_demo_quality.png')
#
# plt.figure(figsize=(8, 8))
#
# plt.plot(size_values, vifp_values, label='VIFP')
# plt.plot(size_values, ssim_values, label='SSIM')
# # plt.plot(size_values, label='NIQE')
# plt.plot(size_values, reco_values, label='RECO')
# plt.plot(size_values, numpy.asarray(psnr_values)/100.0, label='PSNR/100')
# plt.legend(loc='lower right')
# plt.xlabel('JPEG File Size, KB')
# plt.ylabel('Metric')
# plt.savefig('jpg_demo_size.png')

