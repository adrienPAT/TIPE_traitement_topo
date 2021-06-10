import random
import numpy as np
#import mnist_loader
from emnist import extract_training_samples
from PIL import Image
import matplotlib.pyplot as plt
import time

temps = time.clock()
#############################################
#Constantes
#############################################

net_layers = [784, 40, 40, 40, 26]
n_epochs = 500
n_mini_batch_size = 40
eta_c = 0.1

nombre_net = 1

#############################################
#Network functions
#############################################

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                #print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
                print(self.evaluate(test_data))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#############################################
#data functions
#############################################

def vectorized_result(j):
    e = np.zeros((26, 1))
    e[j-1] = 1.0
    return e

def data_wrapper(data, lab):
    inputs = [np.reshape(x, (784,1)) for x in data]
    return [[inputs[i], vectorized_result(lab[i])] for i in range(len(inputs))]

def test_data(data, n):
    k = len(data)
    return [data[random.randint(0, k-1)] for _ in range(n)]
#############################################
#traitement topologique
#############################################



global delta #précision pour les angles
delta = 0.01

global search_size #zone autour de chaque pixel où on considère les autres pixels
search_size = 10

global threshold_number
threshold_number = 6

global threshold #limite pour qu'un pixel soit considéré
threshold = 0.75

global threshold_droites #limite pour qu'une droite soit considérée
threshold_droites = 40

global threshold_longeur_seg #limite pour qu'un segment soit considérée
threshold_longeur_seg = 15

global threshold_fusion_seg #limite pour que 2 seg soient considéerés identiques
threshold_fusion_seg = 15

global threshold_fusion_points #limite pour que 2 seg soient considéerés identiques
threshold_fusion_points = 0.2

global threshold_scan #limite pour qu'on considère une nouvelle lettre
threshold_scan = 4.2069

global x_origin
x_origin = 28
global y_origin
y_origin = 28





def to_deg (angle):
    #Returns: l'angle en degré corresondant à l'indice dans la matrice des droites
    return angle*delta*180 / np.pi

def mat_to_coord(i, j):
    #Returns: coordonée d'un point de la matrice dans le repère centré sur x_origin, y_origin
    # ! on prend l'opposé pour les y pour que (y > 0) <=> (point au dessus de l'axe)
    return (i-x_origin, y_origin-j)
    
def coord_to_mat(x, y):
    #Returns: coordonée d'un point du repère dans la matrice
    #print("coord: ({},{})   origin:({},{})".format(x, y, x_origin, y_origin))
    return (int(round(x+x_origin)), int(round(y_origin-y)))

def eq_droite(coord1, coord2):
    #Returns: (r, t) les coordonées polaires de l'intersection de la droite et de sa perpendiculaire passant par l'origine

    x1, y1 = coord1
    x2, y2 = coord2

    a = x2 - x1
    b = y2 - y1

    #cas particuliers
    if a == 0:#ligne verticale
        if x1 > 0:
            return (abs(x1), 0)
        else:
            return (abs(x1), round(np.pi/delta))
    elif b == 0:#ligne horizontale
        if y1 > 0:
            return (abs(y1), round(np.pi/delta/2))
        else:
            return (abs(y1), -round(np.pi/delta/2))

    c = y1 - b/a * x1

    x_inter = -c / (a/b + b/a)
    y_inter = -a/b * x_inter

    r = int(round(np.sqrt(x_inter**2 + y_inter**2)))
    t = (np.arctan(b/a))/delta

    if y_inter > 0:
        t = int(round(t + np.pi/2/delta))
    else:
        t = int(round(t - np.pi/2/delta))
    
    return (r, t)

def droites_point(i, j, image, coord_droites):
    #calcule les droites possible avec les points proches du point i, j dans la matrice image

    for i_autre in range(i-search_size, i+search_size+1):
        for j_autre in range(j-search_size, j+search_size+1):
            if (j_autre != j or i_autre != i) and image[i_autre, j_autre] > threshold:
                r, t = eq_droite(mat_to_coord(i, j), mat_to_coord(i_autre, j_autre))
                #print(r, t, i, j, i_autre, j_autre)
                coord_droites[int(r), t] += image[i, j] * image[i_autre, j_autre]

def hough(image):
    #calcule toutes les droites sur une image

    w, h = image.shape

    r_max = int(np.sqrt(w**2 + h**2)/2) * 2
    
    #print(w, h, r_max)
    
    coord_droites = np.zeros((r_max, int(2*np.pi/delta)))

    for i in range(search_size, w-search_size):
        for j in range(search_size, h-search_size):

            if image[i, j] > threshold:
                droites_point(i, j, image, coord_droites)

    return coord_droites

class Segment:
    def __init__(self, coord1, coord2, p):
        self.x1, self.y1 = coord1
        self.x2, self.y2 = coord2
        self.p = p #poids

def get_segments(r, t, segments, image):
    # trouve les différents segments qui se trouvent sur une droite et les ajoute à segments
    
    w, h = image.shape
        
    x, y = round(r*np.cos(t*delta)), round(r*np.sin(t*delta))

    x_start, y_start = None, None
    found = False
    
    direction = t*delta - np.pi/2
    
    while abs(x) < w/2 - 2 and abs(y) < h/2 - 2: #on place le curseur à une extremité de l'image
        x -= np.cos(direction)
        y -= np.sin(direction)
    
    x += np.cos(direction)
    y += np.sin(direction)
    
    p = 0
    
    while abs(x) < w/2 - 2 and abs(y) < h/2 - 2:# on parcourt la droite
        x += np.cos(direction)
        y += np.sin(direction)
        x_r = int(round(x))
        y_r = int(round(y))
        #print("coords: ({},{})   matrice: ({})".format(x_r, y_r, coord_to_mat(x_r, y_r)))
        matx, maty = coord_to_mat(x_r, y_r)
        matx = int(matx)
        maty = int(maty)
        pixel = image[matx, maty]
        if pixel >= threshold and x_start == None and y_start == None:#On rencontre le début d'un segment
            x_start, y_start = x_r, y_r
            p += pixel
        elif pixel >= threshold:
            p += pixel
        elif pixel < threshold and x_start != None and y_start != None:#on arrive à la fin du segment
            if np.sqrt((x_start - x_r)**2 + (y_start - y_r)**2) >= threshold_longeur_seg:
                segments.append(Segment((x_start, y_start), (x_r, y_r), p))
                p = 0
            x_start, y_start = None, None

def droites_to_seg(coord_droites, image):

    w, h = image.shape

    r_max = int(np.sqrt(w**2 + h**2)/2)

    segments = []

    for r in range(r_max): #On récupère les droites importantes
        for t in range(int(2*np.pi/delta)):
            if coord_droites[r, t] > threshold_droites:
                get_segments(r, t, segments, image)
    
    return segments

def fusion_segments(segments, a, b):
    pa = segments[a].p
    pb = segments[b].p
    if pa*pb != 0:
        xa1, ya1, xa2, ya2 = segments[a].x1, segments[a].y1, segments[a].x2, segments[a].y2
        xb1, yb1, xb2, yb2 = segments[b].x1, segments[b].y1, segments[b].x2, segments[b].y2

        d = ((xa1-xb1)**2 + (ya1-yb1)**2)**0.5 + ((xa2-xb2)**2 + (ya2-yb2)**2)**0.5

        if d < threshold_fusion_seg:
            segments[a].x1, segments[a].y1, segments[a].x2, segments[a].y2 = (xa1+xb1)/2, (ya1+yb1)/2, (xa2+xb2)/2, (ya2+yb2)/2
            segments[a].p = max(pa, pb)
            segments[b].p = 0

def simplifier_segments(segments):
    n = len(segments)
    for i in range(n):
        for j in range(i):
            fusion_segments(segments, i, j)
        
    for i in range(n-1, -1, -1):
        if segments[i].p == 0:
            del segments[i]

def display_segments(segments):
    for seg in segments:
        plt.plot([seg.x1, seg.x2], [seg.y1, seg.y2])
    plt.show()

def normalisation(segs):
    if segs:
        maxX, minX = segs[0].x1, segs[0].x1
        maxY, minY = segs[0].y1, segs[0].y1
        for segment in segs:
            maxX = max(maxX, segment.x1, segment.x2)
            minX = min(minX, segment.x1, segment.x2)
            maxY = max(maxY, segment.y1, segment.y2)
            minY = min(minY, segment.y1, segment.y2)
        
        for segment in segs:
            segment.x1 = (segment.x1 - minX) / (maxX - minX+1)
            segment.x2 = (segment.x2 - minX) / (maxX - minX+1)
            segment.y1 = (segment.y1 - minY) / (maxY - minY+1)
            segment.y2 = (segment.y2 - minY) / (maxY - minY+1)
    
def extremites(segs):
    points = [(0, 0)] * len(segs) * 2
    for i in range(len(segs)):
        points[2*i] = (segs[i].x1, segs[i].y1)
        points[2*i+1] = (segs[i].x2, segs[i].y2)
    return points

def fusion_points(points, i, j):
    x1, y1 = points[i]
    x2, y2 = points[j]
    d = ((x1-x2)**2 + (y1-y2)**2)**0.5
    if d < threshold_fusion_points:
        points[i] = (69, 420)
        points[j] = ((x1+x2)/2, (y1+y2)/2)

def simplifier_points(points):
    n = len(points)
    for i in range(n):
        for j in range(i):
            fusion_points(points, i, j)
        
    for i in range(n-1, -1, -1):
        if points[i] == (69, 420):
            del points[i]

def li_count(points, minY, maxY):
    c = 0
    for (x, y) in points:
        if y >= minY and y <= maxY:
            c += 1
    return c

def col_count(points, minX, maxX):
    c = 0
    for (x, y) in points:
        if x >= minX and x <= maxX:
            c += 1
    return c

def zone_count(points, X, Y, r):
    c = 0
    for (x, y) in points:
        d = ((x-X)**2 + (y-Y)**2)**0.5
        if d <= r:
            c += 1
    return c

def connected_h(segments, min_Y, max_Y):
    for s in segments:
        if s.y1 >= min_Y and s.y1 <= max_Y and s.y2 >= min_Y and s.y2 <= max_Y and ((s.x1 <= 0.4 and s.x2 >= 0.6) or (s.x2 <= 0.4 and s.x1 >= 0.6)):
            return 1
    return 0

def connected_v(segments, min_X, max_X):
    for s in segments:
        if s.x1 >= min_X and s.x1 <= max_X and s.x2 >= min_X and s.x2 <= max_X and ((s.y1 <= 0.4 and s.y2 >= 0.6) or (s.y2 <= 0.4 and s.y1 >= 0.6)):
            return 1
    return 0

def fill (image, X, Y, color):
    w, h = np.shape(image)
    image[X, Y] = color
    changed = True
    while changed:
        changed = False
        for x in range(w):
            for y in range(h):
                if image[x, y] == 0:
                    if x > 0 and image[x-1, y] == color:
                        image[x, y] = color
                        changed = True
                    elif x < w-1 and image[x+1, y] == color:
                        image[x, y] = color
                        changed = True
                    elif y > 0 and image[x, y-1] == color:
                        image[x, y] = color
                        changed = True
                    elif y < h-1 and image[x, y+1] == color:
                        image[x, y] = color
                        changed = True

def composantes_conexes(image, largeur = 5):
    w, h = np.shape(image)
    image_bis = np.zeros((w, h))
    
    lar1 = round(largeur/2)-1
    lar2 = round(largeur/2)
    
    #on récupère une image où les traits sont un peut plus épais
    for x in range(w):
        for y in range(h):
            if image[x, y] >= threshold:
                for i in range(-lar1, lar2):
                    for j in range(-lar1, lar2):
                        if x+i >= 0 and x+i < w and y+j >= 0 and y+j < h:
                            image_bis[x+i, y+j] = 1
    
    
    fill(image_bis, 0, 0, -1)
    fill(image_bis, w-1, 0, -1)
    fill(image_bis, 0, h-1, -1)
    fill(image_bis, w-1, h-1, -1)
    
    trous = 0
    
    for x in range(w):
        for y in range(h):
            if image_bis[x, y] == 0:
                trous += 1
                fill(image_bis, x, y, trous+1)
    
    #plt.matshow(image_bis)
    
    return trous

def file_to_map(fichier):
    img_file = Image.open(fichier)
    img_file = img_file.convert("L")

    w, h = img_file.size

    image = np.zeros((w, h))

    for x in range(w):
        for y in range(h):
            image[x, y] = 1 - (img_file.getpixel((x, y)) / 255)
    
    return image
    


def criteres(image):

    w, h = np.shape(image)
    
    global x_origin
    global y_origin
    
    x_origin = np.floor(w/2)
    y_origin = np.floor(h/2)
    #print("origin : ({},{})".format(x_origin, y_origin))
    
    #plt.matshow(np.transpose(image))
    #plt.show()

    droites = hough(image)

    segs = droites_to_seg(droites, image)
    simplifier_segments(segs)
    normalisation(segs)

    #display_segments(segs)
    
    p = extremites(segs)
    simplifier_points(p)
    
    p = np.asarray(p)
    #plt.scatter(p[:,0], p[:,1])
    #plt.show()
    
    
    return (col_count(p, 0.8, 1), #droite
            zone_count(p, 1, 1, 0.25),
            li_count(p, 0.8, 1), #haut
            zone_count(p, 0, 1, 0.25),
            col_count(p, 0, 0.2), #gauche
            zone_count(p, 0, 0, 0.25),
            li_count(p, 0, 0.2), #bas
            zone_count(p, 1, 0, 0.25),
            connected_v(segs, 0.75, 1),#trait à droite
            connected_h(segs, 0.75, 1),#trait en haut
            connected_v(segs, 0, 0.25),#trait à gauche
            connected_h(segs, 0, 0.25),#trait en bas
            composantes_conexes(image))#trous


def closest_letter(crit, alphabet):
    d, hd, h, hg, g, bg, b, bd, td, th, tg, tb, cc = crit
    best = alphabet[0]
    
    for l in alphabet:
        if l.dist(d, hd, h, hg, g, bg, b, bd, td, th, tg, tb, cc) < best.dist(d, hd, h, hg, g, bg, b, bd, td, th, tg, tb, cc):
            best = l
    
    return best

def get_letter(image, alphabet):
    return closest_letter(criteres(image), alphabet).char


class Lettre:
    def __init__(self, char, d, hd, h, hg, g, bg, b, bd, td, th, tg, tb, cc):
        self.char = char
        
        self.d = d
        self.hd = hd
        self.h = h
        self.hg = hg
        self.g = g
        self.bg = bg
        self.b = b
        self.bd = bd
        self.td = td
        self.th = th
        self.tg = tg
        self.tb = tb
        self.cc = cc
        #-1: undefined
        #-2: au moins 1
        self.params = (d != -1) + (hd != -1) + (h != -1) + (hg != -1) + (g != -1) + (bg != -1) + (b != -1) + (bd != -1) + (td != -1) + (th != -1) + (tg != -1) + (tb != -1) + (cc != -1)
    
    def dist(self, d, hd, h, hg, g, bg, b, bd, td, th, tg, tb, cc):
        s = 0
        
        if self.d == -2:
            d = -1 - (d != 0)
        if self.h == -2:
            h = -1 - (h != 0)
        if self.g == -2:
            g = -1 - (g != 0)
        if self.b == -2:
            b = -1 - (b != 0)
        
        if self.d != -1:
            s += (d - self.d)**2
        if self.hd != -1:
            s += (hd - self.hd)**2
        if self.h != -1:
            s += (h - self.h)**2
        if self.hg != -1:
            s += (hg - self.hg)**2
        if self.g != -1:
            s += (g - self.g)**2
        if self.bg != -1:
            s += (bg - self.bg)**2
        if self.b != -1:
            s += (b - self.b)**2
        if self.bd != -1:
            s += (bd - self.bd)**2
        if self.td != -1:
            s += (td - self.td)**2
        if self.th != -1:
            s += (th - self.th)**2
        if self.tg != -1:
            s += (tg - self.tg)**2
        if self.tb != -1:
            s += (tb - self.tb)**2
        if self.cc != -1:
            s += (cc - self.cc)**2
        
        s /= self.params
        
        return s**0.5

def separate(image):

    w, h = np.shape(image)
    
    i = 0
    
    letters = []
    
    scanning_letter = False
    
    while i < w-1:
        
        while not scanning_letter and i < w-1:
            i += 1
            scanning_letter = (np.sum(image[i, :]) >= threshold_scan)
        
        i_left = i
        while scanning_letter and i < w-1:
            i += 1
            scanning_letter = (np.sum(image[i, :]) >= threshold_scan)
            if not scanning_letter:
                letters.append((i_left, i))
        
    
    return letters


def criteres_list(image, l_list):
    w, h = np.shape(image)
    lettres_criteres = []
    for left, right in l_list:
        left = max(0, left-2)
        right = min(w-1, right+2)
        sub = np.copy(image[left:right, :])
        lettres_criteres.append(criteres(sub))
    return lettres_criteres


def get_word(image):
    l_list = separate(image)
    CL = criteres_list(image, l_list)
    word = ""
    for crit in CL:
        word += closest_letter(crit, alphabet).char
    return word

#############################################
#data
#############################################

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#training_data = list(training_data)
#validation_data = list(validation_data)
#test_data = list(test_data)

#images_emnist, labels = extract_training_samples('letters')





#############################################
#network
#############################################
##net = Network([784, 30, 26])
###net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
##net.SGD(training_data, 30, 10, 3.0)



##def net(train_data, p, q, layers, epochs, mini_batch_size, eta, test_data = None):
##    k = len(train_data)
##    data = train_data[:k*p//q]
##    net = Network(layers)
##    net.SGD(data, epochs, mini_batch_size, eta, test_data)


##for net_size in range(3, 5):
##    for n_neurons in [30, 40, 50]:
##        print("nombre layers = {}, nombre neurones par layer = {}".format(net_size, n_neurons))
##        net_layers = [n_neurons]*(net_size + 2)
##        net_layers[0] = 784
##        net_layers[net_size + 1] = 26
##        
##        net = Network(net_layers)
##        net.SGD(training_data, n_epochs, n_mini_batch_size, eta_c, test_data)

##net = Network(net_layers)
##net.SGD(training_data, n_epochs, n_mini_batch_size, eta_c, testing_data)



images_emnist, labels = extract_training_samples('letters')
images_emnist, labels = images_emnist[:24000], labels[:24000]

cur = open("tipe_machine_learning_topo_traitement.txt", "r")
rajout_input = cur.readlines()

for i in range(len(rajout_input)):
    rajout_input[i] = list(rajout_input[i][1:-2].split())






training_data = data_wrapper(images_emnist, labels)
for i in range(len(rajout_input)):
    for j in range(len(rajout_input[i])):
        training_data[i][0][j] = rajout_input[i][j]

try_data = test_data(training_data, 10000)

net = Network(net_layers)
net.SGD(training_data, n_epochs, n_mini_batch_size, eta_c, try_data)


print(time.clock - temps)







