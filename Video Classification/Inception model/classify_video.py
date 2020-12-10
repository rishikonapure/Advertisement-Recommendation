
import tensorflow as tf
import sys
import os
import cv2
import math
  
# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
  
label_lines = [line.rstrip() for line
                   in  tf.io.gfile.GFile("tf_files/retrained_labels.txt")]
  
with  tf.io.gfile.GFile("tf_files/retrained_graph.pb", 'rb') as f:
   
    graph_def =  tf.compat.v1.GraphDef()   ## The graph-graph_def is a saved copy of a TensorFlow graph;
    graph_def.ParseFromString(f.read()) #Parse serialized protocol buffer data into variable
    _ = tf.import_graph_def(graph_def, name='') # import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor
      
video_path = sys.argv[1]
writer = None
# classify.py for video processing.

#############################################################
with  tf.compat.v1.Session() as sess:
  
    video_capture = cv2.VideoCapture(video_path)
    i = 0
    while True:  # fps._numFrames < 120
        frame = video_capture.read()[1] # get current frame
        frameId = video_capture.get(1) #current frame number
        i = i + 1
        cv2.imwrite(filename="screens/"+str(i)+"alpha.png", img=frame); # write frame image to file
        image_data = tf.gfile.FastGFile("screens/"+str(i)+"alpha.png", 'rb').read() # get this image file
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})     # analyse the image
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        pos = 1
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            cv2.putText(frame, '%s (score = %.5f)' % (human_string, score), (40, 40 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255))
            print('%s (score = %.5f)' % (human_string, score))
            pos = pos + 1
        print ("\n\n")
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter("recognized.avi", fourcc, 10,
                (frame.shape[1], frame.shape[0]), True)
  
        # write the output frame to disk
        writer.write(frame)
        cv2.imshow("image", frame)  # show frame in window
        cv2.waitKey(1)  # wait 1ms -> 0 until key input
    writer.release()
    video_capture.release()
    cv2.destroyAllWindows()