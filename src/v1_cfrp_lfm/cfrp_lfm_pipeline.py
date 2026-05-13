import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import linregress
from scipy.signal import detrend  
import os
import pickle

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'v1')

cap = cv.VideoCapture(os.path.join(DATA_DIR, 'cropped_cfrp_lfm_og.avi'))
FRAMES = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

def plotPixelIntensity(x,y):
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'correlation_output.mp4'))
    intensityValues = []

    while(True):
        ret, frame = cap.read()

        if not ret:
            break
        else:
            #cropped_frame = frame[0:50, 0:50]
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            intensityValues.append(gray_frame[y][x])
    
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
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'output_correct.mp4'))
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
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'cropped_cfrp_lfm.avi'))
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cropped_frame = frame[y1:y2, x1:x2]
        gray_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
        cv.imshow('Cropped Frame', gray_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    menu()

def plotDetrendedPixelIntensity(x,y):
    # Plot the detrended values
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    plt.plot(detrendPixelIntensities(frames, x, y))
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
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'cropped_cfrp_lfm_og.avi'))
    # 3D array filled with zeroes
    detrended3Dmatrix = np.zeros((FRAMES, HEIGHT, WIDTH), dtype=np.float64)


    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #cropped_frame = frame[60:70, 90:100]
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
        x = int(input("Enter the x-coordinate of the pixel (from 0 to 168): "))
        y = int(input("Enter the y-coordinate of the pixel (from 0 to 288): "))
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
#DETRENDEDMATRIX = displayDetrendedVideo('4')

def displayFourierTransform(display_or_save):
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'output_correct.mp4'))
    ret, frame = cap.read()
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
        out = cv.VideoWriter(os.path.join(DATA_DIR, 'fourier_output0360.mp4'), fourcc, 25.0, (WIDTH-1, HEIGHT-1), isColor=False)

        normalized_fourier3Dmatrix = cv.normalize(fourier3Dmatrix, None, 0, 360, cv.NORM_MINMAX)
        normalized_fourier3Dmatrix = normalized_fourier3Dmatrix.astype(np.uint8)

        for i in range(FRAMES):
            frame = normalized_fourier3Dmatrix[i]
            out.write(frame)

        out.release()
        cap.release()
        cv.destroyAllWindows()
    elif display_or_save == '3':
        #plot
        x = int(input("Enter the x-coordinate of the pixel (from 0 to 168): "))
        y = int(input("Enter the y-coordinate of the pixel (from 0 to 288): "))
        plt.plot(fourier3Dmatrix[:, x, y])
        plt.xlabel('Frame Number')
        plt.ylabel('Fourier Transform Intensity')
        plt.title(f'Fourier Transform Variation of Pixel ({x}, {y})')        
        plt.show()



def displayCorrelationMatrix(display_or_save):
    cap = cv.VideoCapture(os.path.join(DATA_DIR, 'output_correct.mp4'))
    ret, frame = cap.read()
    video3Dmatrix = DETRENDEDMATRIX #np.zeros((FRAMES, HEIGHT-1, WIDTH-1), dtype=np.float64)
    correlation3Dmatrix = np.zeros((FRAMES, HEIGHT-1, WIDTH-1), dtype=np.float64)

    # i = 0
    # while ret:
    #     video3Dmatrix[i] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #     ret, frame = cap.read()
    #     i += 1

    for y in range(HEIGHT-1):
        for x in range(WIDTH-1):
            correlation3Dmatrix[:, y, x] = np.correlate(video3Dmatrix[:, y, x], video3Dmatrix[:, 100, 100], "same")
            print(y, x)
    
    #

    if display_or_save == '1':
        normalized_correlation3Dmatrix = cv.normalize(correlation3Dmatrix, None, 0, 255, cv.NORM_MINMAX)
        normalized_correlation3Dmatrix = normalized_correlation3Dmatrix.astype(np.uint8)
        for i in range(FRAMES):
            cv.imshow('Correlation Matrix Video', correlation3Dmatrix[i])
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    elif display_or_save == '2':
        normalized_correlation3Dmatrix = cv.normalize(correlation3Dmatrix, None, 0, 255, cv.NORM_MINMAX)
        normalized_correlation3Dmatrix = normalized_correlation3Dmatrix.astype(np.uint8)
        fourcc = cv.VideoWriter_fourcc(*'mp4v') 
        out = cv.VideoWriter(os.path.join(DATA_DIR, 'correlation_output.mp4'), fourcc, 25.0, (WIDTH-1, HEIGHT-1), isColor=False)

        for i in range(FRAMES):
            frame = normalized_correlation3Dmatrix[i]
            out.write(frame)

        out.release()
        cap.release()
        cv.destroyAllWindows()
    elif display_or_save == '3':
        x1 = int(input("Enter the x-coordinate of the pixel (from 0 to 168): "))
        y1 = int(input("Enter the y-coordinate of the pixel (from 0 to 288): "))
        x2 = int(input("Enter the x-coordinate of the pixel (from 0 to 168): "))
        y2 = int(input("Enter the y-coordinate of the pixel (from 0 to 288): "))
        x3 = int(input("Enter the x-coordinate of the pixel (from 0 to 168): "))
        y3 = int(input("Enter the y-coordinate of the pixel (from 0 to 288): "))
        plt.plot(correlation3Dmatrix[:, y1, x1])
        plt.plot(correlation3Dmatrix[:, y2, x2])
        plt.plot(correlation3Dmatrix[:, y3, x3])
        plt.legend([f'Pixel ({x1}, {y1})', f'Pixel ({x2}, {y2})', f'Pixel ({x3}, {y3})'])
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

croppedCorrMatrix = CORRMATRIX[430:440, :, :]

def displayCroppedCorrelationMatrix(display_or_save):
    if display_or_save == '1':
        for i in range(140):
            cv.imshow('Cropped Correlation Matrix Video', croppedCorrMatrix[i])
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    elif display_or_save == '2':
        normalized_correlation3Dmatrix = cv.normalize(croppedCorrMatrix, None, 0, 255, cv.NORM_MINMAX)
        normalized_correlation3Dmatrix = normalized_correlation3Dmatrix.astype(np.uint8)
        fourcc = cv.VideoWriter_fourcc(*'mp4v') 
        out = cv.VideoWriter(os.path.join(DATA_DIR, 'cropped_correlation_output.mp4'), fourcc, 25.0, (WIDTH-1, HEIGHT-1), isColor=False)

        for i in range(140):
            frame = normalized_correlation3Dmatrix[i]
            out.write(frame)
            out.release()
            cv.destroyAllWindows()

    elif display_or_save == 'fft':
        disc_levels = int(input("Enter the number of discretization levels: "))
        fourier3Dmatrix = np.zeros((disc_levels, HEIGHT-1, WIDTH-1), dtype=np.float64)
        # i = 0
        # while ret:
        #     video3Dmatrix[i] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #     ret, frame = cap.read()
        #     i += 1
        
        for y in range(HEIGHT-1):
            for x in range(WIDTH-1):
                intensityValues = croppedCorrMatrix[:, y, x]
                padded_intensityValues = np.pad(intensityValues, (0, max(0, disc_levels - len(intensityValues))), 'constant', constant_values=(0))
                fourier3Dmatrix[:, y, x] = np.angle(np.fft.fft(padded_intensityValues))
        selection = int(input("save or plot? (1 or 2): "))
        if selection == 1:
            #save
            fourier3Dmatrix = cv.normalize(fourier3Dmatrix, None, 0, 255, cv.NORM_MINMAX)
            fourier3Dmatrix = fourier3Dmatrix.astype(np.uint8)

            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(os.path.join(DATA_DIR, 'disc_fft_cropped_fourier_output.mp4'), fourcc, 25.0, (WIDTH-1, HEIGHT-1), isColor=False)

            for i in range(disc_levels):
                frame = fourier3Dmatrix[i]
                out.write(frame)
        elif selection == 2:
            #x = int(input("Enter the x-coordinate of the pixel (from 0 to 168): "))
            #y = int(input("Enter the y-coordinate of the pixel (from 0 to 288): "))
            plt.plot(fourier3Dmatrix[:, 85, 68])
            plt.plot(fourier3Dmatrix[:, 190, 68])
            plt.plot(fourier3Dmatrix[:, 138, 68])
            plt.legend([f'Pixel (68, 85)', f'Pixel (68, 190)', f'Pixel (68, 138)'])
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

    selection = int(input("Enter your choice: ")) 

    if selection == 1:
        x = int(input("Enter the x-coordinate of the pixel (from 0 to 168): "))
        y = int(input("Enter the y-coordinate of the pixel (from 0 to 288): "))

        if x < 0 or x > 168 or y < 0 or y > 288:
            print("Invalid coordinates. Please ensure the coordinates are within the specified range.")
        else:
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
        x = int(input("Enter the x-coordinate of the pixel (from 0 to 168): "))
        y = int(input("Enter the y-coordinate of the pixel (from 0 to 288): "))

        if x < 0 or x > 168 or y < 0 or y > 288:
            print("Invalid coordinates. Please ensure the coordinates are within the specified range.")
        else:
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

    else:
        print("\n\nInvalid choice\n\n")
        menu()


menu()

# def detrendPixel(x, y):
#     cap = cv.VideoCapture(os.path.join(DATA_DIR, 'cropped_cfrp_lfm.avi'))
#     intensityValues = []

#     while(True):
#         ret, frame = cap.read()

#         if not ret:
#             break
#         else:
#             gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#             intensityValues.append(gray_frame[y][x])
    
#     #linear regression from scipy.stats
#     frameNumbers = np.arange(len(intensityValues))
#     slope, intercept, _, _, _ = linregress(frameNumbers, intensityValues)
#     linear_trend = slope * frameNumbers + intercept

#     # Remove the linear trend from the intensity values
#     detrended_values = intensityValues - linear_trend

#     return detrended_values

# def displayDetrendedVideo():
#     cap = cv.VideoCapture(os.path.join(DATA_DIR, 'cropped_cfrp_lfm_og.avi'))
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cropped_frame = frame[60:70, 90:100]
#         gray_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
#         double_frame = gray_frame.astype(np.float64) / 255.0
#         frames.append(double_frame)
    
#     cap.release()
    
#     # Detrend each pixel across all frames
#     for y in range(HEIGHT-1):
#         for x in range(WIDTH-1):
#             if x < frames[0].shape[1] and y < frames[0].shape[0]:  # Ensure x and y are within the frame dimensions
#                 detrended_values = detrendPixelIntensities(frames, x, y)
#                 for i in range(FRAMES):
#                     if x < frames[i].shape[1] and y < frames[i].shape[0]:  # Ensure x and y are within the frame dimensions
#                         frames[i][y, x] = detrended_values[i]
#                         print(x, y, i)
    
#     # Define the codec and create a VideoWriter object
#     fourcc = cv.VideoWriter_fourcc(*'XVID')
#     out = cv.VideoWriter('detrended_output.avi', fourcc, 25.0, (10, 10), isColor=False)

#     # Write the detrended frames to the new video file
#     for frame in frames:
#         out.write(frame)

#     # Release the VideoWriter and close all windows
#     out.release()
#     cv.destroyAllWindows()
#     menu()
#     # for i in range(FRAMES):
#     #     cv.imshow('Detrended Video', frames[i])
#     #     if cv.waitKey(1) & 0xFF == ord('q'):
#     #         break

#     # cap.release()
#     # cv.destroyAllWindows()
