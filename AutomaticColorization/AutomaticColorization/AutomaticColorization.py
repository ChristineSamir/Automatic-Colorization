import tensorflow as tf
import glob
import os
import cv2
import numpy as np
import PIL.Image
from tkinter import *
from PIL import ImageFilter
from tkinter.filedialog import askopenfilename,askopenfilenames,askdirectory
 
 
RGBimage_list = []
Grayimage_list = []
ImageHeigth = 224
ImageWidth = 224
Input = tf.placeholder(tf.float32, [None, ImageHeigth*ImageWidth])
InputLabel = tf.placeholder(tf.float32, [None,ImageHeigth,ImageWidth,3])
Keep_prob=tf.placeholder(tf.float32)
 
def read_images_in_folder():
    imagePath = glob.glob('TrainImagesFolder/*.jpg')  
    RGBimage_stack = np.array(np.array([np.array(cv2.imread(imagePath[ImageNo])) for ImageNo in range(len(imagePath))]))
    Grayimage_stack = np.array(np.array([np.array(cv2.imread(imagePath[ImageNo],cv2.IMREAD_GRAYSCALE)) for ImageNo in range(len(imagePath))]))
    return RGBimage_stack,Grayimage_stack
 
def resize_image(image_stack, hResize, wResize): 
    im_resized_stack = np.array( [np.array(cv2.resize(img, (hResize, wResize), interpolation=cv2.INTER_CUBIC)) for img in image_stack]) 
    return im_resized_stack
 
def Read_GreyTest_Images_in_folder():
    print("Read Grey Test Images")
    imagePath = glob.glob('TestImagesFolder/*.jpg')
    GrayTestimage_stack = np.array( [np.array(cv2.imread(imagePath[ImageNo],cv2.IMREAD_GRAYSCALE)) for ImageNo in range(len(imagePath))] )
    return GrayTestimage_stack
 
 
def Read_GreyTest_Image(BrowseImage):
    print("read image")
    GrayTestimage = np.array(cv2.imread(BrowseImage,cv2.IMREAD_GRAYSCALE))
    return GrayTestimage
 
 
 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
 
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
 
def conv2d(Input,Weight):
  return tf.nn.conv2d(Input,Weight,strides=[1,1,1,1],padding='SAME')
 
def MaxPool2d(ActivationMap):
  return tf.nn.max_pool(ActivationMap,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
 
def convolutional_neural_network(Input):
    weights = {'W_conv1':weight_variable([5,5,1,32]),
                      'W_conv2':weight_variable([5,5,32,64]),
                      'W_conv3':weight_variable([5,5,64,128]),
                      'W_conv4':weight_variable([5,5,128,256]),
                      'W_conv5':weight_variable([5,5,256,3]) }
    biases = {'b_conv1':bias_variable([32]),
                      'b_conv2':bias_variable([64]),
                      'b_conv3':bias_variable([128]),
                      'b_conv4':bias_variable([256]),
                      'b_conv5':bias_variable([3])}
    NormalizedInput=(tf.reshape(Input,shape=[-1,ImageHeigth,ImageWidth,1])-128)/128
    conv1=tf.nn.relu(conv2d(NormalizedInput,weights['W_conv1']) + biases['b_conv1'])
    conv2=tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
    conv3=tf.nn.relu(conv2d(conv2,weights['W_conv3']) + biases['b_conv3'])
    conv3=MaxPool2d(conv3)
    conv4=tf.nn.relu(conv2d(conv3,weights['W_conv4']) + biases['b_conv4'])
    conv5=(conv2d(conv4,weights['W_conv5']) + biases['b_conv5'])
    output=(tf.nn.sigmoid(conv5))*255
    output = tf.image.resize_nearest_neighbor(output,[ImageHeigth,ImageWidth])
    return output
 
def unison_shuffled_copies(RGBimage_list,Grayimage_list):
    assert len(RGBimage_list) == len(Grayimage_list)
    Perm = np.random.permutation(len(RGBimage_list))
    return RGBimage_list[Perm], Grayimage_list[Perm]
 
RGBimage_list, Grayimage_list = read_images_in_folder()
Grayimage_list = Grayimage_list.reshape([-1,ImageHeigth*ImageWidth])
unison_shuffled_copies(RGBimage_list,Grayimage_list)
prediction = convolutional_neural_network(Input)
 
EuclideanDistanceLoss = tf.reduce_mean(tf.reduce_sum(tf.subtract(prediction,InputLabel) ** 2) ** 0.5)
train_step = tf.train.AdamOptimizer(1e-4).minimize(EuclideanDistanceLoss)
 
canvas_width = ImageWidth -5
canvas_height = ImageHeigth -5
Browsefilename =''
 
def GUI():
    obj=Tk()
    obj.state('zoomed')
    obj.title("Automatic Image Colorization")  
    obj.configure(background='sky blue')
    obj.winfo_screenwidth()
 
    Label(obj , text="Input Image" , bg="sky blue",fg="Black" ,width=12,font="Helvetica 11 bold").grid(row=0,column=0) 
    Label(obj , text="Output Image" , bg="sky blue",fg="Black",width=26,height=2,font="Helvetica 11 bold").grid(row=0,column=1)    
 
 
    canvas = Canvas(obj, width=canvas_width, height=canvas_height,bg="black")
    canvas.grid(row=2,column=0,padx=17) 
 
    canvas2 = Canvas(obj, width=canvas_width, height=canvas_height,bg="black")
    canvas2.grid(row=2,column=1) 
 
    LoadAndSaveOneImageButton = Button(obj , text="Load And Save An Image" , bg="black",fg="white" ,width=22,font="Helvetica 10 bold",justify=CENTER,relief=RIDGE)
 
    LoadAndSaveImagesButton = Button(obj , text="Load And Save Images" , bg="black",fg="white" ,width=25,font="Helvetica 10 bold",justify=CENTER,relief=RIDGE)
 
    QuitButton = Button(obj , text="Exit" , bg="black",fg="white",width=10,pady=2,font="Helvetica 13 bold",justify=CENTER,relief=RAISED)
 
    VideoInput = Button(obj , text="Video Input" , bg="black",fg="white",width=10,pady=2,font="Helvetica 13 bold",justify=RIGHT,relief=RAISED)
 
    LoadAndSaveOneImageButton.grid(row=5,column=0,pady=5,padx=1)
    LoadAndSaveOneImageButton.bind('<Button-1>', lambda event: LoadImage(event,root=obj,canvas=canvas,canvas2=canvas2))
 
    LoadAndSaveImagesButton.grid(row=7,column=0,pady=5,padx=1) 
    LoadAndSaveImagesButton.bind('<Button-1>', LoadImages)
 
    QuitButton.grid(row=8,column=1,pady=5)
    QuitButton.bind('<Button-1>', quit)
 
    VideoInput.grid(row=5,column=1,pady=5,padx=1)
    VideoInput.bind('<Button-1>', lambda event: VideoToFrames(event,root=obj,canvas=canvas,canvas2=canvas2))
 
    obj.mainloop() 
 
def LoadImage(event,root,canvas,canvas2):
    canvas.delete("all")
    canvas2.delete("all")
    Browsefilename = askopenfilename()
    if Browsefilename=="":
        return
    TempTestGray =Read_GreyTest_Image(Browsefilename).reshape([-1,ImageHeigth*ImageWidth])
    TestResultOneImage = testNN(TempTestGray)
    im = PIL.Image.open(Browsefilename).convert('L')
    Browsefilename=Browsefilename.replace('.jpg','.ppm')
    im.save(Browsefilename)
 
    img = PhotoImage(file=Browsefilename)
    os.remove(Browsefilename)
    canvas.create_image(0,0, anchor=NW, image=img)
 
    Browsefilename=Browsefilename.replace('.ppm','Result.jpg')
    cv2.imwrite(Browsefilename,TestResultOneImage)
    im = PIL.Image.open(Browsefilename)
    Browsefilename=Browsefilename.replace('.jpg','.ppm')
    im.save(Browsefilename)
 
    img2 = PhotoImage(file=Browsefilename)
    os.remove(Browsefilename)
    canvas2.create_image(0,0, anchor=NW, image=img2)
    root.mainloop() 
 
 
def LoadImages(event):
    Browsefilenames = askopenfilenames()
    if Browsefilenames=="":
        return
 
    for filename in Browsefilenames:
        TempTestGray =Read_GreyTest_Image(filename).reshape([-1,ImageHeigth*ImageWidth])
        TestResultOneImage = testNN(TempTestGray)
        IndexName= filename.rfind('/')
        Name=filename[IndexName+1:]
        directory=filename.replace(Name,'Result')
        NewPath=filename.replace(Name,'Result/'+Name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(NewPath,TestResultOneImage)
 
 
def quit(event):    
    os._exit(0x00B97FB0)
 
 
def testNN(TempTestGray):
    saver = tf.train.Saver()
    c=TempTestGray
    with tf.Session() as sess:
        saver.restore(sess, "E:/gray colorization/model/model.ckpt")   
        Output = sess.run(prediction,feed_dict = {Input:TempTestGray,Keep_prob: 0.5})
        Result = np.floor(Output[0])
    return Result
 
 
 
def trainNN():
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, "ModelFolder/model.ckpt")
        for epoch in range(500):
            EpochLoss = 0
            for batch in range(int(6208/10)):
                print("Batch Num ",batch + 1)
                _, BatchLoss = sess.run([train_step,EuclideanDistanceLoss],feed_dict={Input: Grayimage_list[batch*10:(batch+1)*10], InputLabel: RGBimage_list[batch*10:(batch+1)*10], Keep_prob: 0.5})
                EpochLoss +=BatchLoss
            print("epoch: ",epoch + 1, ",Loss: ",EpochLoss)
            save_path = saver.save(sess, "ModelFolder/model.ckpt")
 
 
def VideoToFrames(event,root,canvas,canvas2):
    Browsefilename = askopenfilename()
    if Browsefilename=="":
        return  
 
    dir_path = Browsefilename
    cap = cv2.VideoCapture(dir_path)
 
    frames=[]
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
            root.delay = 15
 
            root.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            img =PhotoImage(image = root.photo)
            canvas.create_image(0,0, anchor=NW, image=img)
        else:
            print("Bullshit")
            break
    greyImages = []
    counter=0
    for f in frames:
        IndexName= Browsefilename.rfind('/')
        Name=Browsefilename[IndexName+1:]
        directory=Browsefilename.replace(Name,'Result')
        NewPath=Browsefilename.replace(Name,'Result/'+str(counter)+'.jpg')
        counter=counter+1
        fr = cv2.cvtColor(f,cv2.COLOR_RGB2GRAY)
        frame = fr.reshape([-1,ImageHeigth*ImageWidth])
        TestResultOneImage = testNN(frame)
        greyImages.append(TestResultOneImage)
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output='Test Output Video.avi'
    out = cv2.VideoWriter(directory+'/'+output, fourcc, 10.0, (ImageWidth, ImageHeigth),True)
    for image in greyImages:    
        image=image.astype(np.uint8)
        out.write(image)
    out.release()
    print("YAY DONE YO!")
    canvas.delete("all")
    canvas2.delete("all")
    im = PIL.Image.open(Browsefilename).convert('L')
    Browsefilename=Browsefilename.replace('.jpg','.ppm')
    im.save(Browsefilename)
 
 
    img = PhotoImage(file=Browsefilename)
    os.remove(Browsefilename)
    canvas.create_image(0,0, anchor=NW, image=img)
 
    Browsefilename=Browsefilename.replace('.ppm','Result.jpg')
    cv2.imwrite(Browsefilename,TestResultOneImage)
    im = PIL.Image.open(Browsefilename)
    Browsefilename=Browsefilename.replace('.jpg','.ppm')
    im.save(Browsefilename)
 
    img2 = PhotoImage(file=Browsefilename)
    os.remove(Browsefilename)
    canvas2.create_image(0,0, anchor=NW, image=img2)
 
 
GUI()