
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

XML_DIR = "../../data/annotations/xmls/"
IMG_DIR = "../../data/images/"
def listAllFiles(dir):
    files = [f for f in listdir(dir) if isfile(join(dir,f))]
    files_with_dir = list(map(lambda f:dir+f ,files))
    return files_with_dir

def getInfoTupleForXml(xmlFile,imgDir):

    tree = ET.parse(xmlFile)
    root = tree.getroot()
    label = tree.find("object").find('name').text
    size_node= tree.find("size")
    bbox_node = tree.find("object").find('bndbox')
    imgFile = imgDir + tree.find("filename").text
    label_index = -1
    if label=='cat':
        label_index = 0
    elif label=='dog':
        label_index = 1

    if label_index==-1:
        print("something wrong! Not cat nor dog!")
    width= int(size_node.find('width').text)
    height= int(size_node.find('height').text)
    xmin = int(bbox_node.find('xmin').text)
    ymin = int(bbox_node.find('ymin').text)
    xmax = int(bbox_node.find('xmax').text)
    ymax = int(bbox_node.find('ymax').text)
    
    return [xmlFile,imgFile,label_index,(xmin/width*100,ymin/height*100,xmax/width*100,ymax/height*100),(width,height)]

#Main
if __name__ == '__main__':
    xmlFiles = listAllFiles(XML_DIR)
    imgFiles = listAllFiles(IMG_DIR)
    infoList = list(map(lambda f:getInfoTupleForXml(f,IMG_DIR) ,xmlFiles))
    print(imgFiles[0])
    print(infoList[0][1])

