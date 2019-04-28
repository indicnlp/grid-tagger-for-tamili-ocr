
import cv2 
import numpy as np 
from tqdm import tqdm
from glob import glob
import random
import os
import sys
import json
import math
import copy
from pprint import pprint, pformat

import logging
FORMAT_STRING = "%(levelname)-8s:%(name)-8s.%(funcName)-8s>> %(message)s"
logging.basicConfig(format=FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
import sys

from pprint import pprint, pformat

def mkdir_if_exist_not(name):
    if not os.path.isdir(name):
        return os.makedirs(name)

def imshow(name, img, resize_factor = 0.4):
    return cv2.imshow(name,
                      cv2.resize(img,
                                 (0,0),
                                 fx=resize_factor,
                                 fy=resize_factor))

def rotate(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    return outImg

def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' 
    https://stackoverflow.com/questions/13226038/calculating-angle-between-two-lines-in-python
    compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

class Grid:
    STATE_GRAB = 0
    
    STATE_MOVE = 1
    STATE_MOVE_GRID  = 2
    
    STATE_SCALE  = 3
    STATE_SCALE_GRID  = 4
    STATE_SCALE_BOXES  = 5

    STATE_ROTATE  = 6

    BULK_V = -1
    BULK_O =  0
    BULK_H =  1

    BULK = [BULK_O, BULK_H, BULK_V]
    
    def __init__(self, args, name, img,
                 default_box_size= [75, 90],
                 grid_m=11, grid_n=19,
                 box_dist_m=30, box_dist_n=50,
                 unit_rotation = 0.1):

        self.args = args
        self.name = name
        self.filepath = '{}/{}.grid2'.format(self.args.prefix, self.name)
        
        self.img = rotate(img, 90)
        self.source = self.img.copy()
        self.source_backup = self.img.copy()
        
        #should saved to file
        self.finished = False
        self.grid_anchor_point = None
        self.scale_factor = int(1/0.3)
        self.default_box_size = default_box_size
        self.default_box_size_backup = default_box_size
        self.rotation = 0
        
        self.    grid_m, self.    grid_n =     grid_m,     grid_n
        self.box_dist_m, self.box_dist_n = box_dist_m, box_dist_n

        self.sentinel = [[0, 0], [0, 0]]
        self.boxes = [ self.sentinel, ]

        # state trackers
        self.move_first_point_set = False
        self.move_grid_first_point_set = False
        
        self.scale_first_point_set = False
        self.scale_grid_first_point_set = False
        self.scale_boxes_first_point_set = False

        self.unit_rotation = unit_rotation
        self.bulk_orientation = self.BULK[0]   # BULK_O
        
        self.grabbed = 0
        self.grabbed_box_backup = None
        
        self.grabbed_bulk = []
        self.grabbed_bulk_box_backup = []

        
        cv2.namedWindow(self.name)
        cv2.namedWindow(self.name + '.slice')
        cv2.setMouseCallback(self.name, self.callback)

        self.load_state()

    def imshow(self, name, img):
        imshow(name, img, 1.0/self.scale_factor)
        
    def print_state(self):
        log.debug('state: {}'.format(self.state))
        log.debug('boxes[0]:{}'.format(pformat(self.boxes[0])))
        log.debug('grabbed: id: {} bkup: {} box:{}'.format(self.grabbed,
                                                           pformat(self.grabbed_box_backup),
                                                           pformat(self.boxes[self.grabbed])
        ))



    def generate_new_grid(self, anchor, box_dist_m=None, box_dist_n=None):
        
        self.boxes = [self.sentinel,]
        self.grid_anchor_point = anchor
        if not box_dist_m:  box_dist_m = self.box_dist_m
        if not box_dist_n:  box_dist_n = self.box_dist_n

        for i in range(self.grid_m):
            boxes_row = []
            for j in range(self.grid_n):
                box = [
                        [
                            anchor[0] + j * int(2 * self.default_box_size[0] + box_dist_m),
                            anchor[1] + i * int(2 * self.default_box_size[0] + box_dist_n)
                        ],
                        self.default_box_size
                ]
                
                self.boxes.append(box)
                
        log.debug('generated {} boxes grid'.format(len(self.boxes)))

        self.box_dist_m, self.box_dist_n = box_dist_m, box_dist_n
        
    def distance(self, a, b):
        return math.sqrt(
            ((a[0] - b[0]) ** 2)
            + ((a[1] - b[1]) ** 2)
        )
    

    def find_closest_box(self, point):
        point_dists = [(b, self.distance(point, b[0])) for b in self.boxes ]
        point_dists = sorted(point_dists, key=lambda x: x[1])
        return point_dists[0]

            
    def callback(self, event, x, y, flags, param):
        temp = [self.scale_factor * x, self.scale_factor * y]

        if event == cv2.EVENT_LBUTTONDOWN:
            log.info('temp: {}'.format(pformat(temp)))

            if self.state == self.STATE_GRAB:
                cbox, distance = self.find_closest_box(temp)
                if distance < max(self.default_box_size[0], cbox[1]):
                    
                    log.debug('cbox: ' + pformat(cbox))
                    
                    self.grabbed = self.boxes.index(cbox)
                    self.grabbed_box_backup = copy.copy(self.boxes[self.grabbed])
                    
                    log.debug('cbox, cbox index, cbox distance : {}, {}, {}'.format(
                        cbox, self.grabbed, distance))

                    
                    if self.BULK[self.bulk_orientation] == self.BULK_H:
                        grabbed = (self.grabbed - 1) // self.grid_n
                        self.grabbed_bulk_box_backup = []
                        self.grabbed_bulk = [grabbed * self.grid_n + i
                                             for i in list(range(1,
                                                                 self.grid_n + 1))]

                        self.grabbed = 0
                        self.grabbed_box_backup = None
                                  
                    elif self.BULK[self.bulk_orientation] == self.BULK_V:
                        grabbed = self.grabbed  % self.grid_n
                        if grabbed == 0:
                            grabbed = self.grid_n
                            
                        self.grabbed_bulk = [grabbed + i
                                             for i in list(range(0,
                                                                 len(self.boxes) - self.grid_n + 1,
                                                                 self.grid_n))]
                        self.grabbed = 0
                        self.grabbed_box_backup = None
                        
                    self.grabbed_bulk_box_backup = []
                    log.debug('grabbed {} boxes in bulk: {}'.format(len(self.grabbed_bulk_box_backup),
                                                                       pformat(self.grabbed_bulk)))
                    for i in self.grabbed_bulk:
                        log.debug('grabbing {}nth box in bulk'.format(i))
                        self.grabbed_bulk_box_backup.append(
                            copy.copy(self.boxes[i]))

                    log.debug('grabbed {} boxes in bulk: {}'.format(len(self.grabbed_bulk_box_backup),
                                                                    pformat(self.grabbed_bulk)))

            if self.state == self.STATE_SCALE_GRID:
                self.generate_new_grid(temp)
                    
        if event == cv2.EVENT_MOUSEMOVE:
            
            if self.state == self.STATE_MOVE:

                if self.grabbed != 0 or self.grabbed_bulk != []:                 
                    if not self.move_first_point_set :
                        self.move_first_point = temp
                        self.move_first_point_set = True
                        log.debug('move first point: {},{}'.format(*temp))
                    else:
                        dx = temp[0] - self.move_first_point[0]
                        dy = temp[1] - self.move_first_point[1]
                        log.debug('moving by {}, {}'.format(dx, dy))

                        if self.BULK[self.bulk_orientation] == self.BULK_O:
                            (x, y) = self.grabbed_box_backup[0]
                            log.debug('moving box {}'.format(self.grabbed))
                            self.boxes[self.grabbed][0] = [x + dx, y + dy]
                        else:
                            for i in range(len(self.grabbed_bulk)):
                                log.debug('moving {}nth {} box in bulk'.format(i, self.grabbed + i))
                                (x, y) = self.grabbed_bulk_box_backup[i][0]
                                self.boxes[ self.grabbed_bulk[i] ][0] = [x + dx, y + dy]

            if self.state == self.STATE_MOVE_GRID:
                if not self.move_grid_first_point_set :
                    self.deferer = 0
                    self.move_grid_first_point = temp
                    self.move_grid_first_point_set = True
                    log.debug('move first point: {},{}'.format(*temp))
                else:
                    self.deferer += 1
                    if not self.deferer % 10:
                        dx = temp[0] - self.move_grid_first_point[0]
                        dy = temp[1] - self.move_grid_first_point[1]
                        dx, dy = int(dx//10), int(dy//10)
                        #log.debug('moving by {}, {}'.format(dx, dy))
                        for i in range(len(self.boxes)):
                            (x, y) = self.grid_backup[i][0]
                            #log.debug('moving box {}'.format(i))
                            self.boxes[i][0] = [x + dx, y + dy]

                    
            if self.state == self.STATE_SCALE:

                if self.grabbed != 0:
                    if not self.scale_first_point_set :
                        self.scale_first_point = temp
                        self.scale_first_point_set = True
                    else:
                        angle = get_angle( self.scale_first_point,
                                           self.boxes[self.grabbed][0], temp )

                        if self.bulk_orientation == self.BULK_O:
                            self.boxes[self.grabbed][1] = [int(self.grabbed_box_backup[1][0]
                                                               -
                                                               0.1 * angle * self.grabbed_box_backup[1][0])
                                                           
                                                           ,int(self.grabbed_box_backup[1][1]
                                                                -
                                                                0.1 * angle * self.grabbed_box_backup[1][1])
                            ]

                    self.print_state()

            if self.state == self.STATE_SCALE_GRID:
                if not self.scale_grid_first_point_set :
                    self.scale_grid_first_point = temp
                    self.scale_grid_first_point_set = True
                    log.debug('scale grid first point: {},{}'.format(*temp))
                else:
                    dx = temp[0] - self.scale_grid_first_point[0]
                    dy = temp[1] - self.scale_grid_first_point[1]
                    self.generate_new_grid(self.scale_grid_first_point, 0.1 * dx,  0.1 * dy)
                    
                self.draw()

                
            if self.state == self.STATE_SCALE_BOXES:
                if not self.scale_boxes_first_point_set :
                    self.scale_boxes_first_point = temp
                    self.scale_boxes_first_point_set = True
                    log.debug('scale boxes first point: {},{}'.format(*temp))
                else:
                    angle = get_angle( self.scale_boxes_first_point,
                                       [0,0], temp )
                    
                    self.default_box_size = [int(self.default_box_size_backup[0]
                                                 -
                                                 0.1 * angle * self.default_box_size_backup[0])
                                             
                                             ,int(self.default_box_size_backup[1]
                                                  -
                                                  0.1 * angle * self.default_box_size_backup[1])
                    ]
                    print(angle, self.default_box_size)
                    
                    self.generate_new_grid(self.grid_anchor_point)
                    
                self.draw()
                
        if event == cv2.EVENT_LBUTTONUP:
            log.info('temp: {}'.format(pformat(temp)))
            self.state = self.STATE_GRAB
            
            if self.state == self.STATE_MOVE:
                log.debug('moved box {} to {}'.format(self.grabbed, pformat(temp)))
                self.move_first_point_set = False
                    
            if self.state == self.STATE_SCALE:
                self.scale_first_point_set = False
                if self.grabbed != 0:
                    self.grabbed_box_backup = copy.copy(self.boxes[self.grabbed])
                    
                    
            
        if event == cv2.EVENT_LBUTTONDBLCLK:
            log.info('temp: {}'.format(pformat(temp)))

            #self.generate_new_grid(temp)
            self.print_state()

        self.draw()
        self.imshow(self.name, self.img)
        if self.grabbed:
            self.imshow(self.name + '.slice',
                   cv2.resize(self.get_box_pixels(self.boxes[self.grabbed]),
                              (0,0), fx=5, fy=5 )
            )
            
        if self.grabbed_bulk:
            imgs = [self.get_box_pixels(self.boxes[i]) for i in self.grabbed_bulk]
            if self.BULK[self.bulk_orientation] == self.BULK_H:
                imgs = np.hstack(imgs)
            elif self.BULK[self.bulk_orientation] == self.BULK_V:
                imgs = np.vstack(imgs)

            self.imshow(self.name + '.slice',
                   cv2.resize(
                       imgs,
                       (0,0), fx=1.5, fy=1.5 )
            )


    def get_box_pixels(self, box):
        center, size = box
        x1, x2 = center[0] - size[0], center[0] + size[0]
        y1, y2 = center[1] - size[1], center[1] + size[1]

        return self.source[y1:y2, x1:x2]
        
    def draw_box(self, img, box, color=(255,0,0), thickness=4):
        center, size = box
        x1, x2 = center[0] - size[0], center[0] + size[0]
        y1, y2 = center[1] - size[1], center[1] + size[1]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def draw(self):
        del self.img
        self.img = self.source.copy()
        for box in self.boxes:
            self.draw_box(self.img, box)

        if self.grabbed_box_backup:
            self.draw_box(self.img, self.boxes[self.grabbed], (0,255,0), 6)

        if self.grabbed_bulk_box_backup:
            for box in self.grabbed_bulk:
                self.draw_box(self.img, self.boxes[box], (0,255,0), 6)


    def start(self):
        print('image state == {} and args.force == {}'.format(self.finished, self.args.force))
        if self.finished == False or self.args.force:
            self.draw()
            self.state = self.STATE_GRAB
            self.source = rotate(self.source_backup, self.rotation)
            while True:
                cv2.setMouseCallback(self.name, self.callback)
                self.imshow(self.name, self.img)

                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    log.debug('setting state to grab')
                    self.state = self.STATE_GRAB
                    if self.grabbed != 0 or self.grabbed_bulk != []:                 
                        if self.BULK[self.bulk_orientation] == self.BULK_O:
                            self.boxes[self.grabbed] = self.grabbed_box_backup
                        else:
                            for i in range(len(self.grabbed_bulk)):
                                log.debug('ungrabbing {}nth {} box in bulk'.format(i, self.grabbed + i))
                                self.boxes[self.grabbed_bulk[i]] = self.grabbed_bulk_box_backup[i]

                    if self.default_box_size_backup != None:
                        self.default_box_size = copy.copy(self.default_box_size_backup)
                        self.default_box_size_backup = None

                    self.grabbed = 0
                    self.grabbed_box_backup = None
                    self.grabbed_bulk = []
                    self.grabbed_bulk_box_backup = []
                    self.grid_backup = []

                elif k == ord('q'):
                    log.debug('quit!')
                    exit(0)

                elif k == ord('g'):
                    log.debug('setting state to grab')
                    self.state = self.STATE_GRAB

                elif k == ord('k'):
                    log.debug('setting state to grab')
                    self.state = self.STATE_GRAB
                    self.grabbed += 1
                    if self.grabbed == len (self.boxes):
                        self.grabbed = 1

                    self.draw()

                elif k == ord('s'):
                    log.debug('setting state to scale')
                    self.state = self.STATE_SCALE
                    self.scale_first_point_set = False

                elif k == ord('S'):
                    log.debug('setting state to scale boxes')
                    self.state = self.STATE_SCALE_BOXES
                    self.default_box_size_backup = copy.copy(self.default_box_size)
                    self.scale_boxes_first_point_set = False

                elif k == ord('m'):
                    log.debug('move grabbed box')
                    self.state = self.STATE_MOVE
                    self.move_first_point_set = False

                elif k == ord('n'):
                    log.debug('generate new grid')
                    self.state = self.STATE_SCALE_GRID
                    self.scale_grid_first_point_set = False

                elif k == ord('M'):
                    log.debug('move  grid')
                    self.state = self.STATE_MOVE_GRID
                    self.move_grid_first_point_set = False
                    self.grid_backup = copy.copy(self.boxes)

                elif k == ord('R'):
                    log.debug('rotate image {} degree'.format(self.rotation))
                    self.state = self.STATE_ROTATE
                    self.source = rotate(self.source, self.unit_rotation)
                    self.rotation += self.unit_rotation
                    self.draw()

                elif k == ord('r'):
                    log.debug('rotate image {} degree'.format(-self.rotation))
                    self.state = self.STATE_ROTATE
                    self.source = rotate(self.source, -self.unit_rotation)
                    self.rotation -= self.unit_rotation
                    self.draw()

                elif k == ord('c'):
                    log.debug('clear roration')
                    self.state = self.STATE_ROTATE
                    self.source = self.source_backup.copy()
                    self.rotation = 0
                    self.draw()                

                elif k == ord('b'):
                    self.bulk_orientation += 1
                    self.grabbed_bulk = []
                    self.grabbed_bulk_box_backup = []
                    if self.bulk_orientation > 2:
                        self.bulk_orientation = 0
                    log.debug('setting bulk to {}'.format(self.bulk_orientation))

                elif k == ord(' '):
                    self.save_state()

                elif k == ord('F'):
                    self.finished = True
                    self.save_state()

                elif k == ord('\n'):
                    self.save_state()
                    break


            self.save_state()

        cv2.destroyAllWindows()
        return self.boxes[1:]

    def save_state(self):
        mkdir_if_exist_not('{}'.format(self.args.prefix))
        log.info('saving data to {}'.format(self.filepath))
        with open(self.filepath, 'w') as f:
            f.write(
                json.dumps((
                    self.boxes,
                    self.scale_factor,
                    self.default_box_size,
                    self.unit_rotation,
                    self.rotation,
                    self.grid_m, self.grid_n,
                    self.box_dist_m, self.box_dist_n,
                    self.grid_anchor_point,
                    self.finished,
                ))
            )
            
    def load_state(self):
        try:
            log.info('loading data from {}'.format(self.filepath))
            with open(self.filepath) as f:
                state = json.loads(f.read())
                (self.boxes,
                 self.scale_factor,
                 self.default_box_size,
                 self.unit_rotation,
                 self.rotation,
                 self.grid_m, self.grid_n,
                 self.box_dist_m, self.box_dist_n,
                 self.grid_anchor_point,
                 self.finished) = state
                
        except:
            log.exception('====')


def process(args, filepath=None):
    if filepath == None:
        filepath = args.filepath
        
    source = cv2.imread(filepath)
    cv2.imwrite('source.jpg', source)

    img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    grid = Grid(args, os.path.basename(filepath), source)
    boxes = grid.start()

    if args.very_verbose:
        log.debug(boxes)
        log.debug(len(boxes))


    total_count = 0
    shapes = []
    source = rotate(source, 90 + grid.rotation)
    for i, box in enumerate(boxes):

        try:
            (x, y), (dx, dy) = box
            char = source [
                y-dy : y+dy,
                x-dx : x+dx
            ]
        
            total_count += 1
            shapes.append(char)
            print(char.shape)
            if args.very_verbose:
                cv2.imshow('char', char)
                cv2.waitKey(0)
                
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:            
            log.exception('#####')
            
    return total_count, shapes
    


def write_shapes(args, shapes):
    for i, shape in enumerate(shapes):
        mkdir_if_exist_not('{}/{}'.format(args.prefix, i))
        #newimg = cv2.resize(shape, (args.size, args.size))
        newimg = shape
        if args.verbose:
            cv2.imshow("result", newimg)
            key = cv2.waitKey(0)
            
        cv2.imwrite('{}/{}/{}'.format(args.prefix, i, os.path.basename(args.filepath)), newimg)

import argparse
if __name__ == '__main__':

    filepath = random.choice(glob('sheets/*.jpg'))
    
    parser = argparse.ArgumentParser(description='Grid-segmenter')
    parser.add_argument('-f','--filepath',
                        help='path to the image file',
                        default=filepath, dest='filepath')

    
    parser.add_argument('-t','--type',
                        help='type of interface 0 for point based and 1 for line based',
                        default=0, dest='type', type=int)
    
    parser.add_argument('-d','--prefix-dir',
                        help='path to the image file',
                        default='sliced', dest='prefix')

    parser.add_argument('-s','--size',
                        help='size of the resulting shape',
                        default=120, dest='size')
    
    parser.add_argument('-v', '--verbose',
                        help='shows all the grid overlayed in input image',
                        action='store_true', default=False, dest='verbose')

    parser.add_argument('-V', '--very-verbose',
                        help='shows all the pieces of the characters',
                        action='store_true', default=False, dest='very_verbose')
        
    args = parser.parse_args()

    pprint(args)
    total_count, shapes = process(args)
    print('total count: {}'.format(total_count))
    write_shapes(args, shapes)
