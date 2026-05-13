import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import linregress
from scipy.signal import detrend  
import os
import pickle

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'v3')

cap = cv.VideoCapture(os.path.join(DATA_DIR, 'new_gfrp_2000_flux.avi'))
FRAMES = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
CORRSIZE = 2*FRAMES - 1

def plotPixelIntensity(x,y):
    cap = cv.VideoCapture(os.path.join(DATA_DIR, '160ogdisc_fourcorr.mp4'))
    intensityValues = []


    while(True):
        ret, frame = cap.read()

        if not ret:
            break
        else:
            #cropped_frame = frame[0:50, 0:50]
            #gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
            intensityValues.append(frame[115][115])
    
    # intensityValues = np.interp(intensityValues, (np.min(intensityValues), np.max(intensityValues)), (-360, 360))

    plt.plot(intensityValues)
    plt.xlabel('Frame Number')
    plt.ylabel('Pixel Intensity')
    plt.title(f'Intensity Variation at Pixel ({x}, {y})')
    # plt.ylim([0, 360]) 
    plt.show()
    cv.waitKey(0) 
    cv.destroyAllWindows()

def displayFrame(frameNumber):
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'fourier_output_norm.mp4'))
    cap.set(cv.CAP_PROP_POS_FRAMES, frameNumber - 1)
    ret, frame = cap.read()
    if not ret:
        print("Frame not found")
    else: 
        #gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        plt.imshow(frame)#)cv.imshow('Selected frame', gray_frame)
        plt.show()
        cv.waitKey(0) 
        cv.destroyAllWindows()
    
    menu()

def cropVideo(x1, y1, x2, y2):
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'new_gfrp_2000_flux.avi'))

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(os.path.join(DATA_DIR, 'cropped_output.avi'), fourcc, 25.0, (573, 459), isColor=False)
    for i in range(FRAMES):
        ret, frame = cap.read()

        if not ret:
            break

        cropped_frame = frame[y1:y2, x1:x2]
        gray_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)

        out.write(gray_frame)
        cv.imshow('Cropped Frame', gray_frame)

    cap.release()
    out.release()
    cv.waitKey(0)
    cv.destroyAllWindows()

   # menu()

def plotDetrendedPixelIntensity(x,y):
    # Plot the detrended values
    # frames = []
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #     frames.append(gray_frame)
    plt.plot(DETRENDEDMATRIX[:,y,x])
    plt.xlabel('Frame Number')
    plt.ylabel('Detrended Pixel Intensity')
    plt.title(f'Detrended Intensity Variation at Pixel ({x}, {y})')
    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()

def detrendPixelIntensities(frames, x, y):
    intensity_values = []
    for frame in frames:
        intensity_values.append(frame[y, x])
    
    # # Calculate the linear trend
    frame_numbers = np.arange(len(intensity_values))
    slope, intercept, _, _, _ = linregress(frame_numbers, intensity_values)
    linear_trend = slope * frame_numbers + intercept
    
    # Detrend the intensity values
    detrended_values = intensity_values - linear_trend
    
    return detrended_values


def displayDetrendedVideo(display_or_save):
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'new_gfrp_2000_flux.avi'))
    # 3D array filled with zeroes
    detrended3Dmatrix = np.zeros((FRAMES, HEIGHT, WIDTH), dtype=np.float64)


    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        double_frame = gray_frame.astype(np.float64) / 255.0
        frames.append(double_frame)


    for y in range(HEIGHT):
        for x in range(WIDTH):
            detrended3Dmatrix[:, y, x] = detrendPixelIntensities(frames, x, y)
            print(y, x)


    
    # plt.plot(normalized_detrended3Dmatrix[:, 0, 0])
    # plt.show() 
            
    if display_or_save == '1':
        normalized_detrended3Dmatrix = cv.normalize(detrended3Dmatrix, None, 0, 255, cv.NORM_MINMAX)
        normalized_detrended3Dmatrix = normalized_detrended3Dmatrix.astype(np.uint8)        
        for i in range(FRAMES):
            cv.imshow('Detrended Video', normalized_detrended3Dmatrix[i])
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    elif display_or_save == '2':
        normalized_detrended3Dmatrix = cv.normalize(detrended3Dmatrix, None, 0, 255, cv.NORM_MINMAX)
        normalized_detrended3Dmatrix = normalized_detrended3Dmatrix.astype(np.uint8)        
        fourcc = cv.VideoWriter_fourcc(*'mp4v') 
        out = cv.VideoWriter(os.path.join(DATA_DIR, 'output_correct.mp4'), fourcc, 25.0, (WIDTH, HEIGHT), isColor=False)

        for i in range(FRAMES):
            frame = normalized_detrended3Dmatrix[i]
            #frame = detrended3Dmatrix[:, :, i]        
            out.write(frame)

        out.release()
        cap.release()
        cv.destroyAllWindows()

    elif display_or_save == '3':
        x = int(input("Enter the x-coordinate of the pixel: "))
        y = int(input("Enter the y-coordinate of the pixel: "))
        plt.plot(detrended3Dmatrix[:, x, y])
        plt.xlabel('Frame Number')
        plt.ylabel('Detrended Pixel Intensity')
        plt.title(f'Detrended Intensity Variation at Pixel ({x}, {y})')
        plt.show() 
    
    elif display_or_save == '4':
        return detrended3Dmatrix

CONSTANT_FILE = os.path.join(DATA_DIR, 'detrended_matrix.pkl')

if os.path.exists(CONSTANT_FILE):
    with open(CONSTANT_FILE, 'rb') as f:
        DETRENDEDMATRIX = pickle.load(f)
else:
    DETRENDEDMATRIX = displayDetrendedVideo('4')
    with open(CONSTANT_FILE, 'wb') as f:
        pickle.dump(DETRENDEDMATRIX, f)

def displayFourierTransform(display_or_save):
    # cap = cv.VideoCapture(os.path.join(DATA_DIR, 'output_correct.mp4'))
    # ret, frame = cap.read()
    video3Dmatrix = DETRENDEDMATRIX #np.zeros((FRAMES, HEIGHT-1, WIDTH-1), dtype=np.float64)
    fourier3Dmatrix = np.zeros((FRAMES, HEIGHT-1, WIDTH-1), dtype=np.float64)

    # i = 0
    # while ret:
    #     video3Dmatrix[i] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #     ret, frame = cap.read()
    #     i += 1

    for y in range(HEIGHT-1):
        for x in range(WIDTH-1):
            intensityValues = video3Dmatrix[:, y, x]
            fourier3Dmatrix[:, y, x] = np.angle(np.fft.fft(intensityValues))
            print(y, x)
    

    # Display 
    if display_or_save == '1':
        for i in range(FRAMES):
            cv.imshow('Fourier angle Video', fourier3Dmatrix[i])
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    elif display_or_save == '2':
        fourcc = cv.VideoWriter_fourcc(*'mp4v') 
        out = cv.VideoWriter(os.path.join(DATA_DIR, 'fourier_output_norm.mp4'), fourcc, 25.0, (WIDTH-1, HEIGHT-1), isColor=False)

        normalized_fourier3Dmatrix = cv.normalize(fourier3Dmatrix, None, 0, 255, cv.NORM_MINMAX)
        normalized_fourier3Dmatrix = normalized_fourier3Dmatrix.astype(np.uint8)

        for i in range(FRAMES):
            frame = normalized_fourier3Dmatrix[i]
            out.write(frame)

        out.release()
        cap.release()
        cv.destroyAllWindows()
    elif display_or_save == '3':
        #plot
        x = int(input("Enter the x-coordinate of the pixel: "))
        y = int(input("Enter the y-coordinate of the pixel: "))
        plt.plot(fourier3Dmatrix[:, x, y])
        plt.xlabel('Frame Number')
        plt.ylabel('Fourier Transform Intensity')
        plt.title(f'Fourier Transform Variation of Pixel ({x}, {y})')        
        plt.show()



def displayCorrelationMatrix(display_or_save): #higer frame rate 2000 frames ask ishant bhaiya
    # cap = cv.VideoCapture(os.path.join(DATA_DIR, 'new_gfrp_2000_flux.avi'))
    # ret, frame = cap.read()
    video3Dmatrix = DETRENDEDMATRIX #.reshape((FRAMES, HEIGHT*WIDTH)) #np.zeros((FRAMES, HEIGHT-1, WIDTH-1), dtype=np.float64)
    # #video2Dmatrix = video2Dmatrix.T
    
    correlation3Dmatrix = np.zeros((CORRSIZE, HEIGHT-1, WIDTH-1), dtype=np.float64)

    # i = 0
    # while ret:
    #     video3Dmatrix[i] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #     ret, frame = cap.read()
    #     i += 1 
    

    # for y in range(HEIGHT-1):
    #     for x in range(WIDTH-1):
    #         correlation3Dmatrix[:, y, x] = np.correlate(video3Dmatrix[:, y, x], video3Dmatrix[:, 200, 200], "full")
    #         print(y, x)

    if display_or_save == '1':
        #normalized_correlation3Dmatrix = cv.normalize(correlation3Dmatrix, None, 0, 255, cv.NORM_MINMAX)
        #normalized_correlation3Dmatrix = normalized_correlation3Dmatrix.astype(np.uint8)
        for i in range(CORRSIZE):
            cv.imshow('Correlation Matrix Video', CORRMATRIX[i])
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    elif display_or_save == '2':
        normalized_correlation3Dmatrix = cv.normalize(CORRMATRIX, None, 0, 255, cv.NORM_MINMAX)
        normalized_correlation3Dmatrix = normalized_correlation3Dmatrix.astype(np.uint8)
        fourcc = cv.VideoWriter_fourcc(*'mp4v') 
        out = cv.VideoWriter(os.path.join(DATA_DIR, 'correlation_output.mp4'), fourcc, 25.0, (WIDTH-1, HEIGHT-1), isColor=False)

        for i in range(CORRSIZE):
            frame = normalized_correlation3Dmatrix[i]
            out.write(frame)

        out.release()
        cap.release()
        cv.destroyAllWindows()
    elif display_or_save == '3':
        x1 = int(input("Enter the x-coordinate of the pixel: "))
        y1 = int(input("Enter the y-coordinate of the pixel: "))

        x2 = int(input("Enter the x-coordinate of the pixel: "))
        y2 = int(input("Enter the y-coordinate of the pixel: "))

        x3 = int(input("Enter the x-coordinate of the pixel: "))
        y3 = int(input("Enter the y-coordinate of the pixel: "))

        x4 = int(input("Enter the x-coordinate of the pixel: "))
        y4 = int(input("Enter the y-coordinate of the pixel: "))

        # x5 = int(input("Enter the x-coordinate of the pixel: "))
        # y5 = int(input("Enter the y-coordinate of the pixel: "))

        # x6 = int(input("Enter the x-coordinate of the pixel: "))
        # y6 = int(input("Enter the y-coordinate of the pixel: "))

        # x7 = int(input("Enter the x-coordinate of the pixel: "))
        # y7 = int(input("Enter the y-coordinate of the pixel: "))

        # x8 = int(input("Enter the x-coordinate of the pixel: "))
        # y8 = int(input("Enter the y-coordinate of the pixel: "))

        # x9 = int(input("Enter the x-coordinate of the pixel: "))
        # y9 = int(input("Enter the y-coordinate of the pixel: "))

        # x10 = int(input("Enter the x-coordinate of the pixel: "))
        # y10 = int(input("Enter the y-coordinate of the pixel: "))

        # x11 = int(input("Enter the x-coordinate of the pixel: "))
        # y11 = int(input("Enter the y-coordinate of the pixel: "))

        # x12 = int(input("Enter the x-coordinate of the pixel: "))
        # y12 = int(input("Enter the y-coordinate of the pixel: "))

        # x13 = int(input("Enter the x-coordinate of the pixel: "))
        # y13 = int(input("Enter the y-coordinate of the pixel: "))

        plt.plot(CORRMATRIX[:, y1, x1])
        plt.plot(CORRMATRIX[:, y2, x2])
        plt.plot(CORRMATRIX[:, y3, x3])
        plt.plot(CORRMATRIX[:, y4, x4])
        # plt.plot(CORRMATRIX[:, y5, x5])
        # plt.plot(CORRMATRIX[:, y6, x6])
        # plt.plot(CORRMATRIX[:, y7, x7])
        # plt.plot(CORRMATRIX[:, y8, x8])
        # plt.plot(CORRMATRIX[:, y9, x9])
        # plt.plot(CORRMATRIX[:, y10, x10])
        # plt.plot(CORRMATRIX[:, y11, x11])
        # plt.plot(CORRMATRIX[:, y12, x12])
        # plt.plot(CORRMATRIX[:, y13, x13])

        plt.title('Correlation Transform Variation at Pixels')
        plt.legend([f'Pixel ({x1}, {y1})', f'Pixel ({x2}, {y2})', f'Pixel ({x3}, {y3})' 
                    , f'Pixel ({x4}, {y4})' , 
                    # f'Pixel ({x5}, {y5})' , f'Pixel ({x6}, {y6})' 
                    # , f'Pixel ({x7}, {y7})' , f'Pixel ({x8}, {y8})' , f'Pixel ({x9}, {y9})' 
                    # , f'Pixel ({x10}, {y10})' , f'Pixel ({x11}, {y11})' , f'Pixel ({x12}, {y12})' , 
                    f'Sound Pixel'])
        plt.xlabel('Frame Number')
        plt.ylabel('Correlation Transform Variation')
        #plt.title(f'Correlation Transform Variation at Pixel ({x}, {y})')        
        plt.show()
    elif display_or_save == '4':
        return correlation3Dmatrix

CORR_FILE = os.path.join(DATA_DIR, 'correlation_matrix.pkl')

if os.path.exists(CORR_FILE):
    with open(CORR_FILE, 'rb') as f:
        CORRMATRIX = pickle.load(f)
else:
    CORRMATRIX = displayCorrelationMatrix('4')
    with open(CORR_FILE, 'wb') as f:
        pickle.dump(CORRMATRIX, f)

croppedCorrMatrix = CORRMATRIX[930:1090, :, :]

def displayCroppedCorrelationMatrix(display_or_save):
    if display_or_save == '1':
        for i in range(FRAMES):
            cv.imshow('Cropped Correlation Matrix Video', croppedCorrMatrix[i])
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    elif display_or_save == '2':
        normalized_correlation3Dmatrix = cv.normalize(croppedCorrMatrix, None, 0, 255, cv.NORM_MINMAX)
        normalized_correlation3Dmatrix = normalized_correlation3Dmatrix.astype(np.uint8)
        fourcc = cv.VideoWriter_fourcc(*'mp4v') 
        out = cv.VideoWriter(os.path.join(DATA_DIR, 'cropped_correlation_output.mp4'), fourcc, 25.0, (WIDTH-1, HEIGHT-1), isColor=False)

        for i in range(CORRSIZE):
            frame = normalized_correlation3Dmatrix[i]
            out.write(frame)
        out.release() 
        cv.destroyAllWindows()

    elif display_or_save == 'fft':
       # disc_levels = int(input("Enter the number of discretization levels: ")) 
        fourier3Dmatrix = np.zeros((160, HEIGHT-1, WIDTH-1), dtype=np.float64)
        # i = 0
        # while ret:
        #     video3Dmatrix[i] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #     ret, frame = cap.read() 
        #     i += 1 
        
        for y in range(HEIGHT-1):
            for x in range(WIDTH-1):
                intensityValues = croppedCorrMatrix[:, y, x]
                # padded_intensityValues = np.pad(intensityValues, (0, max(0, disc_levels - len(intensityValues))), 'constant', constant_values=(0))
                fourier3Dmatrix[:, y, x] = np.angle(np.fft.fft(intensityValues))
        selection = int(input("save or plot? (1 or 2): "))
        if selection == 1:
            #save
            fourier3Dmatrix = cv.normalize(fourier3Dmatrix, None, 0, 255, cv.NORM_MINMAX)
            fourier3Dmatrix = fourier3Dmatrix.astype(np.uint8)

            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(os.path.join(DATA_DIR, 'disc_fft_cropped_fourier_output.mp4'), fourcc, 25.0, (WIDTH-1, HEIGHT-1), isColor=False)

            for i in range(160):
                frame = fourier3Dmatrix[i]
                out.write(frame)
        elif selection == 2:
            #x = int(input("Enter the x-coordinate of the pixel: ")) 
            #y = int(input("Enter the y-coordinate of the pixel: "))
            plt.plot(fourier3Dmatrix[:, 115, 115])
            plt.plot(fourier3Dmatrix[:, 230, 115])
            plt.plot(fourier3Dmatrix[:, 345, 115])

            # plt.plot(fourier3Dmatrix[:, 230, 115])
            # plt.plot(fourier3Dmatrix[:, 230, 230])
            # plt.plot(fourier3Dmatrix[:, 230, 345]) 

            # plt.plot(fourier3Dmatrix[:, 345, 115])
            # plt.plot(fourier3Dmatrix[:, 345, 230])
            # plt.plot(fourier3Dmatrix[:, 345, 345])

            # plt.plot(fourier3Dmatrix[:, 460, 115])
            # plt.plot(fourier3Dmatrix[:, 460, 230])
            # plt.plot(fourier3Dmatrix[:, 460, 345])

            plt.plot(fourier3Dmatrix[:, 200, 200])
            plt.legend([f'defect (115, 115)', f'defect (115, 230)', f'defect (115, 345)'
                        # , f'sound (230, 115)', f'sound (230, 230)', f'sound (230, 345)'
                        # , f'sound (345, 115)', f'sound (345, 230)', f'sound (345, 345)'
                        # , f'sound (460, 115)', f'sound (460, 230)', f'sound (460, 345)'
                        , f'sound (10, 10)' ])
            plt.show()
    # Three graphs to be plotted 
    # graph 1 - plot ratio of main lobe to first side lobe to the right
    # 20log(ratio) - y axis
    # depth - x axis 
    # graph 2 - phase of fft against depths
    # graph 3 - phase of fft of corr main lobe against depths
    # problem - accessing the first lobe to the right is proving to be difficult 
    # since it is not always the second highest value
def graph2():
    fourier3Dmatrix = np.zeros((160, HEIGHT-1, WIDTH-1), dtype=np.float64)
    for y in range(HEIGHT-1):
        for x in range(WIDTH-1):
            intensityValues = croppedCorrMatrix[:, y, x]
            fourier3Dmatrix[:, y, x] = np.angle(np.fft.fft(intensityValues))
    x = int(input("Enter the x-coordinate of the pixel: ")) 
    y = int(input("Enter the y-coordinate of the pixel: "))
    plt.plot(fourier3Dmatrix[:, y, x])
    plt.show()

def graph3():
    fourier3Dmatrix = np.zeros((160, HEIGHT-1, WIDTH-1), dtype=np.float64)
    for y in range(HEIGHT-1):
        for x in range(WIDTH-1):
            intensityValues = croppedCorrMatrix[:, y, x]
            fourier3Dmatrix[:, y, x] = np.angle(np.fft.fft(intensityValues))
    x = int(input("Enter the x-coordinate of the pixel: ")) 
    y = int(input("Enter the y-coordinate of the pixel: "))
    plt.plot(fourier3Dmatrix[:, y, x])
    plt.show()
# main menu 
def menu():
    print("******************** MENU ********************")
    print("Press 1 to view the intensity variation graph of a specific pixel")
    print("Press 2 to view a specific frame of a video")
    print("Press 3 to view a cropped portion of the video")
    print("Press 4 to view the detrended intensity variation graph of a specific pixel")
    print("Press 5 to view the detrended video")
    print("Press 6 to view the Fourier Transform of the video")
    print("Press 7 to view the Correlation Transform of the video")
    print("Press 8 to view the cropped Correlation Transform of the video")
    print("Press 9 to crop the main lobe of the video")

    selection = int(input("Enter your choice: ")) 

    if selection == 1:
        x = int(input("Enter the x-coordinate of the pixel: "))
        y = int(input("Enter the y-coordinate of the pixel: "))

        # if x < 0 or x > 168 or y < 0 or y > 288:
        #     print("Invalid coordinates. Please ensure the coordinates are within the specified range.")
        # else: 
        plotPixelIntensity(x,y)
        
        

    elif selection == 2:
        frameNumber = int(input("Enter the frame number to view (from 1 to 1000): "))
        if frameNumber < 1 or frameNumber > 1000:
            print("Invalid frame number. Please enter a frame number between 1 and 1000.")
            menu()
        else:
            displayFrame(frameNumber)
        

    elif selection == 3:
        x1 = int(input("Enter the x-coordinate of the top-left corner of the cropped portion (from 1 to 169): "))
        y1 = int(input("Enter the y-coordinate of the top-left corner of the cropped portion (from 1 to 289): "))
        x2 = int(input("Enter the x-coordinate of the bottom-right corner of the cropped portion (from 1 to 169): "))
        y2 = int(input("Enter the y-coordinate of the bottom-right corner of the cropped portion (from 1 to 289): "))    

        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0 or x1 >= x2 or y1 >= y2:
            print("Invalid coordinates. Please ensure the coordinates are within the specified range and x2 > x1 and y2 > y1.")
            menu()
        else:
            cropVideo(x1, y1, x2, y2)

    elif selection == 4:
        x = int(input("Enter the x-coordinate of the pixel: "))
        y = int(input("Enter the y-coordinate of the pixel: "))

        # if x < 0 or x > 168 or y < 0 or y > 288:
        #     print("Invalid coordinates. Please ensure the coordinates are within the specified range.")
        # else: 
        plotDetrendedPixelIntensity(x,y)

    elif selection == 5:
        display_or_save = input("1 to display, 2 to save, and 3 to plot: ")
        displayDetrendedVideo(display_or_save)

    elif selection == 6:
        display_or_save = input("1 to display, 2 to save, and 3 to plot: ")
        displayFourierTransform(display_or_save)
    
    elif selection == 7:
        display_or_save = input("1 to display, 2 to save, and 3 to plot: ")
        displayCorrelationMatrix(display_or_save)
    
    elif selection == 8:
        display_or_save = input("1 to display, 2 to save, fft to take fourier trsfm: ")
        displayCroppedCorrelationMatrix(display_or_save)
    
    elif selection == 9:
        graph1()

    else:
        print("\n\nInvalid choice\n\n")
        menu()


menu()