import cv2
import glob
path = '/imagenet'
path2 = '/data'

#file1 = open(path + 'imglist_train.txt','r')
#count = 0
#sep = ' '

#while True:
    #line = file1.readline()
    #stripped = line.split(sep, 1)[0]
    #count +=1
    #if not line:
    #    break
    #print(stripped)
    #image = cv2.imread(path2+stripped)
    #cv2.imwrite('data/train/'+str(count)+'.jpg',image)
    #if count == 800: break
#file1.close()

paths = []
paths = glob.glob('C:/Users/Konrad/Desktop/IO_Duszki/imagenet/train/*.jpeg')
print(paths)
count = 0
while True:
    image = cv2.imread(str(paths[count]))

    if count < 5000:
        cv2.imwrite('./data/test/'+str(count)+'.jpg',image)
    else:
        cv2.imwrite('./data/train/'+str(count)+'.jpg',image)

    count = count + 1
    if count == len(paths): break;