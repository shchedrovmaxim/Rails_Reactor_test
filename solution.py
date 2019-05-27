#!/usr/bin/env python
# coding: utf-8

# In[25]:


from PIL import Image
import os
import argparse
import numpy as np


# In[13]:


image_list = [Image.open(os.path.join('dev_dataset', img)) for img in os.listdir('dev_dataset') if not os.path.isdir(img)]


# In[14]:


def print_name(image1, image2):
    name1 = image1.filename.split(os.sep)[-1]
    name2 = image2.filename.split(os.sep)[-1]
    print(name1, name2)


# In[15]:


def RMS(a, b):
    return np.sqrt(np.power(a - b, 2).mean())


# In[16]:


def hamming_distance(h1, h2):
    return np.sum(h1 != h2)


# In[18]:


def RGB_clasificator(img1,img2):
    """ img1 - first image 
        img2 - second image
        return True if image same or False if diferent
    """
    ds1 = img1.resize((1024, 1024))
    ds2 = img2.resize((1024, 1024))
    data1 = ds1.getdata()
    data2 = ds2.getdata()
    
    r1 = [[d[0], 0, 0] for d in data1]
    g1 = [[0, d[0], 0] for d in data1]
    b1 = [[0, 0, d[0]] for d in data1]

    r2 = [[d[0], 0, 0] for d in data2]
    g2 = [[0, d[0], 0] for d in data2]
    b2 = [[0, 0, d[0]] for d in data2]
    count = 0
    for i in range(len(r1)):
        if (abs(r1[i][0] - r2[i][0]) < 20 and 
            abs(g1[i][1] - g2[i][1]) < 20 and 
            abs(b1[i][2] - b2[i][2]) < 20):
            count += 1 
        else: continue
    score = (count/len(r1))*100
    if score >50:
        return True
    else: 
        return False


# In[19]:


def rule_base_clasificator(img1,img2):
    """ img1 - first image 
        img2 - second image
        return True if image same or False if diferent
    """
    dark1 = img1.convert('L')
    dark2 = img2.convert('L')
    dark1_resize = dark1.resize((1024, 1024))
    dark2_resize = dark2.resize((1024, 1024))

    error = RMS(np.array(dark1_resize), np.array(dark2_resize))
    defirence_size = np.abs(dark1.size[0] / dark1.size[1] - dark2.size[0] / dark2.size[1])
    
    if error < 9.5 and (error == 0 or defirence_size < 0.5):
        return True
    else:
        return False


# In[20]:


def bild_hash(image):
    """Build hash
        If pixel in image bigger than average's all pixels 
        hash append 1 else append 0
        :param image: PIL Image object
    """
    img = image.resize((32, 32)).convert('L')
    pixel_data = np.array(list(img.getdata()))
    hash_im =[]
    avg = np.mean(pixel_data)
    for i in pixel_data:
        if i>avg:
            hash_im.append('1')
        else:
            hash_im.append('0')
        hashe = np.array(hash_im)
    return hashe


# In[39]:


def hash_base_clasificator(img1,img2):
    """ img1 - first image 
        img2 - second image
        return True if image same or False if diferent
    """
    hash1 = bild_hash(img1)
    hash2 = bild_hash(img2)
    
    if hamming_distance(hash1,hash2) < 300:
        return True
    else: 
        return False


# In[27]:


def dir_path(path):
    """
    Check if argument is path.
    
    :param path: path to check
    :returns: path back if it is valid
    :raises ArgumentTypeError: raises if path is not valid
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("directory: '{}' is not valid.".format(path))


# In[26]:


def main(path, method):
    """
    Reads images and perform classification on each pair.
    :param path: path to folder where images are located
    :param method: method to do classification with
    """
    image_list = [Image.open(os.path.join(path, img)) for img in os.listdir(path) if not os.path.isdir(img)]
    classify = {
        'rulebase': rule_base_clasificator,
        'hash': hash_base_clasificator,
        'RGB': RGB_clasificator
    }

    for i in range(len(im_list)):
        for j in range(i + 1, len(im_list)):
            try:
                if classify[method](image_list[i], image_list[j]):
                    print_name(image_list[i], image_list[j])
            except KeyError as e:
                raise argparse.ArgumentTypeError("method: '{}' is not valid.".format(method))

if __name__ == "__main__":
    """
    Argument parser. 
    Parse args and run main method.
    """
    parser = argparse.ArgumentParser(description='First test task on images similarity.')
    parser.add_argument('--path', metavar='PATH', required=True, type=dir_path, help='folder with images')
    parser.add_argument('--method', metavar='METH', default='ensemble', type=str,
                        help="""rule - rule-based method;
                        hash - hash-based method;
                        RGB - RGB comparison method""")
    
    args = parser.parse_args()
    main(args.path, args.method)


# In[ ]:





# In[ ]:





# In[ ]:




