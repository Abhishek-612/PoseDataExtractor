import numpy as np
import cv2
import math,os
import time
from scipy.ndimage.filters import gaussian_filter
from config_reader import config_reader
from model import get_testing_model



def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


tic = 0
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image


def process(input_image, params, model_params,label):
    ''' Start of finding the Key points of full body using Open Pose.'''
    dic = {0: 'Attentive',1: 'Head-rested on hand',2: 'Not looking at the screen',3: 'Writing',4: 'Leaning back'}
    if not os.path.exists('data/'+dic[label]):
        os.mkdir('data/'+dic[label])
    oriImg = input_image  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    for m in range(1):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)
        output_blobs = model.predict(input_img)
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []  # To store all the key points which a re detected.
    peak_counter = 0

    prinfTick(1)  # prints time required till now.

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    prinfTick(2)  # prints time required till now.

    canvas = create_blank(input_image.shape[1],input_image.shape[0])  # B,G,R order

    try:
        cv2.line(canvas, all_peaks[14][0][0:2], all_peaks[16][0][0:2], colors[16], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[14][0][0:2], all_peaks[0][0][0:2], colors[14], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[15][0][0:2], all_peaks[0][0][0:2], colors[3], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[15][0][0:2], all_peaks[17][0][0:2], colors[17], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[1][0][0:2], all_peaks[0][0][0:2], colors[0], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[1][0][0:2], all_peaks[8][0][0:2], colors[8], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[9][0][0:2], all_peaks[8][0][0:2], colors[9], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[9][0][0:2], all_peaks[10][0][0:2], colors[10], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[1][0][0:2], all_peaks[11][0][0:2], colors[11], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[12][0][0:2], all_peaks[11][0][0:2], colors[12], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[12][0][0:2], all_peaks[13][0][0:2], colors[13], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[1][0][0:2], all_peaks[2][0][0:2], colors[12], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[3][0][0:2], all_peaks[2][0][0:2], colors[2], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[3][0][0:2], all_peaks[4][0][0:2], colors[10], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[1][0][0:2], all_peaks[5][0][0:2], colors[9], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[6][0][0:2], all_peaks[5][0][0:2], colors[6], thickness=2)
    except:
        pass
    try:
        cv2.line(canvas, all_peaks[6][0][0:2], all_peaks[7][0][0:2], colors[13], thickness=2)
    except:
        pass

    for i in range(18):  # drawing all the detected key points.
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    global count,name
    cv2.imwrite('data/{}/{}_{}_{}.png'.format(dic[label],name,dic[label],str(count)),canvas)

    getPoints(all_peaks,label,dic)
    print()
    print('Done')
    print()
    return canvas

def getPoints(all_peaks,label,dic):
    try:
        p0 = all_peaks[0][0][:2]
    except:
        p0 = (np.nan,np.nan)
    try:
        p1 = all_peaks[1][0][:2]
    except:
        p1 = (np.nan,np.nan)
    try:
        p2 = all_peaks[2][0][:2]
    except:
        p2 = (np.nan,np.nan)
    try:
        p3 = all_peaks[3][0][:2]
    except:
        p3 = (np.nan,np.nan)
    try:
        p4 = all_peaks[4][0][:2]
    except:
        p4 = (np.nan,np.nan)
    try:
        p5 = all_peaks[5][0][:2]
    except:
        p5 = (np.nan,np.nan)
    try:
        p6 = all_peaks[6][0][:2]
    except:
        p6 = (np.nan,np.nan)
    try:
        p7 = all_peaks[7][0][:2]
    except:
        p7 = (np.nan,np.nan)
    try:
        p8 = all_peaks[8][0][:2]
    except:
        p8 = (np.nan,np.nan)
    try:
        p9 = all_peaks[9][0][:2]
    except:
        p9 = (np.nan,np.nan)
    try:
        p10 = all_peaks[10][0][:2]
    except:
        p10 = (np.nan,np.nan)
    try:
        p11 = all_peaks[11][0][:2]
    except:
        p11 = (np.nan,np.nan)
    try:
        p12 = all_peaks[12][0][:2]
    except:
        p12 = (np.nan,np.nan)
    try:
        p13 = all_peaks[13][0][:2]
    except:
        p13 = (np.nan,np.nan)
    try:
        p14 = all_peaks[14][0][:2]
    except:
        p14 = (np.nan,np.nan)
    try:
        p15 = all_peaks[15][0][:2]
    except:
        p15 = (np.nan,np.nan)
    try:
        p16 = all_peaks[16][0][:2]
    except:
        p16 = (np.nan,np.nan)
    try:
        p17 = all_peaks[17][0][:2]
    except:
        p17 = (np.nan,np.nan)

    global count,name
    f.write('{}_{}_{}.png,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(name,dic[label],count,label,p0[0],p0[1],p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1],p5[0],p5[1],p6[0],p6[1],p7[0],p7[1],p8[0],p8[1],p9[0],p9[1],p10[0],p10[1],p11[0],p11[1],p12[0],p12[1],p13[0],p13[1],p14[0],p14[1],p15[0],p15[1],p16[0],p16[1],p17[0],p17[1]))


def checkPosition(all_peaks):
    try:
        f = 0
        print('Head angle:', round(math.degrees(calcAngle(all_peaks[0][0][0:2], all_peaks[1][0][0:2]))))
        print('Right Ear angle:', round(math.degrees(calcAngle(all_peaks[0][0][0:2], all_peaks[16][0][0:2]))))
        print('Left Ear angle:', round(math.degrees(calcAngle(all_peaks[0][0][0:2], all_peaks[17][0][0:2]))))
        print('Right Shoulder angle:', round(math.degrees(calcAngle(all_peaks[1][0][0:2], all_peaks[2][0][0:2]))))
        print('Left Shoulder angle:', round(math.degrees(calcAngle(all_peaks[1][0][0:2], all_peaks[5][0][0:2]))))
        print('Ear distance:', round(calcDistance(all_peaks[0][0][0:2], all_peaks[17][0][0:2]), 2),
              round(calcDistance(all_peaks[0][0][0:2], all_peaks[16][0][0:2]), 2))
        print('Shoulder distance:', round(calcDistance(all_peaks[1][0][0:2], all_peaks[2][0][0:2]), 2),
              round(calcDistance(all_peaks[1][0][0:2], all_peaks[5][0][0:2]), 2))
        if (all_peaks[16]):
            a = all_peaks[16][0][0:2]  # Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2]  # Left Ear
        b = all_peaks[11][0][0:2]  # Hip
        angle = calcAngle(a, b)
        degrees = round(math.degrees(angle))
        if (f):
            degrees = 180 - degrees
        if (degrees < 70):
            return 1
        elif (degrees > 110):
            return -1
        else:
            return 0
    except Exception as e:
        print("person not in lateral view and unable to detect ears or hip")


def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if (ax == bx):
            return 1.570796
        return math.atan2(by - ay, bx - ax)
    except Exception as e:
        print("unable to calculate angle")


def checkHandFold(all_peaks):
    try:
        if (all_peaks[3][0][0:2]):
            try:
                if (all_peaks[4][0][0:2]):
                    distance = calcDistance(all_peaks[3][0][0:2],
                                            all_peaks[4][0][0:2])  # distance between right arm-joint and right palm.
                    armdist = calcDistance(all_peaks[2][0][0:2],
                                           all_peaks[3][0][0:2])  # distance between left arm-joint and left palm.
                    if (distance < (armdist + 100) and distance > (
                            armdist - 100)):  # this value 100 is arbitary. this shall be replaced with a calculation which can adjust to different sizes of people.
                        print("Not Folding Hands")
                    else:
                        print("Folding Hands")
            except Exception as e:
                print("Folding Hands")
    except Exception as e:
        try:
            if (all_peaks[7][0][0:2]):
                distance = calcDistance(all_peaks[6][0][0:2], all_peaks[7][0][0:2])
                armdist = calcDistance(all_peaks[6][0][0:2], all_peaks[5][0][0:2])
                if (distance < (armdist + 100) and distance > (armdist - 100)):
                    print("Not Folding Hands")
                else:
                    print("Folding Hands")
        except Exception as e:
            print("Unable to detect arm joints")


def calcDistance(a, b):  # calculate distance between two points.
    try:
        x1, y1 = a
        x2, y2 = b
        return math.hypot(x2 - x1, y2 - y1)
    except Exception as e:
        print("unable to calculate distance")


def prinfTick(i):
    toc = time.time()
    print('processing time%d is %.5f' % (i, toc - tic))


if __name__ == '__main__':
    # import pandas as pd
    # df = pd.read_csv('pose_data.csv')
    # print(df['p0'])

    tic = time.time()
    # print('start processing...')
    model = get_testing_model()
    model.load_weights('model.h5')

    print('\n========================================================================')
    global name
    name = input('Enter your name: ')

    if not os.path.exists('data'):
        os.mkdir('data')


    f = open('data/pose_data.csv','a')
    if not os.path.exists('pose_data.csv'):
        f.write('name,label,p0_x,p0_y,p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,p4_x,p4_y,p5_x,p5_y,p6_x,p6_y,p7_x,p7_y,p8_x,p8_y,p9_x,p9_y,p10_x,p10_y,p11_x,p11_y,p12_x,p12_y,p13_x,p13_y,p14_x,p14_y,p15_x,p15_y,p16_x,p16_y,p17_x,p17_y\n')

    print('\n========================================================================')
    print('0: Attentive\n1: Head-rested on hand\n2: Not looking at the screen\n3: Writing\n4: Leaning back')
    print('========================================================================')
    choice = int(input('Pick one and remember to do the action related to THAT ACTION ONLY! :)\n(P.S. - Any other number to exit)\n'))

    print('\n*********  BE READY! (Recording will go for 2 minutes)  *********\n')

    while 0 <= choice < 4:
        cap = cv2.VideoCapture(0)
        vi = cap.isOpened()
        start = time.time()
        global count
        count = 0
        if (vi == True):
            cap.set(100, 160)
            cap.set(200, 120)
            time.sleep(0.2)

            while (time.time() - start) <= 120:
                tic = time.time()
                print('Time:{}\n'.format(str(time.time()-start)))
                ret, frame = cap.read()
                params, model_params = config_reader()
                count+=1
                canvas = process(frame, params, model_params,choice)
                cv2.imshow("capture", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
        else:
            print("unable to open camera")

        cv2.destroyAllWindows()

        print()
        print('0: Attentive\n1: Head-rested on hand\n2: Not looking at the screen\n3: Writing\n4: Leaning back')
        print('========================================================================')
        choice = int(input('Pick one and remember to do the action related to THAT ACTION ONLY! :)\n(P.S. - Any other number to exit)\n'))


f.close()