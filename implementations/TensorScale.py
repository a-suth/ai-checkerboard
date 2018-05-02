import os, csv
import tensorflow as tf
from PIL import Image
import pandas as pd
from implementations.Implementation import Implementation

import numpy as np

class TensorScale(Implementation):
    def __init__(self):
        self.data = []
        self.data_heading = ['ABOVE', 'BELOW', 'LEFT', 'RIGHT', 'CENTRE']
        self.csv_filename = 'data.csv'
        self.batch_size = 100
        self.train_steps = 1000

    def generate_data_csv(self):
        print('loading files...')
        for filename in os.listdir('./trainingimgs'):
            with Image.open(filename) as img:
                self.imgdata = np.asarray(img)   
                self.imgdata.flags.writeable = False

                for x in range(0, len(self.imgdata)):       # for each row
                    row = self.imgdata[x]
                    for y in range(0, len(row)):
                        above,below,left,right = self.get_surrounding(x,y)
                        if None in [above,below,left,right]:
                            pass
                        else:
                            self.data.append([self.imgdata[above[0],above[1],0],self.imgdata[below[0],below[1],0],self.imgdata[left[0],left[1],0],self.imgdata[right[0],right[1],0],self.imgdata[x,y,0]])
                            self.data.append([self.imgdata[above[0],above[1],1],self.imgdata[below[0],below[1],1],self.imgdata[left[0],left[1],1],self.imgdata[right[0],right[1],1],self.imgdata[x,y,1]])
                            self.data.append([self.imgdata[above[0],above[1],2],self.imgdata[below[0],below[1],2],self.imgdata[left[0],left[1],2],self.imgdata[right[0],right[1],2],self.imgdata[x,y,2]])


                            #self.data.append(np.array([self.imgdata[above[0],above[1],:3],self.imgdata[below[0],below[1],:3],self.imgdata[left[0],left[1],:3],self.imgdata[right[0],right[1],:3],self.imgdata[x,y,:3]]))

    def save_to_csv(self):
        print('writing to %s...' % self.csv_filename)
        with open(self.csv_filename, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(self.data_heading)
            for line in self.data:
                writer.writerow(line)
        print('done')

    def load_data(self):
        label_name = 'CENTRE'
        train = pd.read_csv(filepath_or_buffer=self.csv_filename,
                            names=self.data_heading,  # list of column names
                            dtype=int,
                            header=0  # ignore the first row of the CSV file.
                           )
        # train now holds a pandas DataFrame, which is data structure
        # analogous to a table.

        # 1. Assign the DataFrame's labels (the right-most column) to train_label.
        # 2. Delete (pop) the labels from the DataFrame.
        # 3. Assign the remainder of the DataFrame to train_features
        train_features, train_label = train, train.pop(label_name)

        test = pd.read_csv(filepath_or_buffer=self.csv_filename,
                            names=self.data_heading,  # list of column names
                            header=0  # ignore the first row of the CSV file.
                           )
        test_features, test_label = test, test.pop(label_name)

        # Return four DataFrames.
        return (train_features, train_label), (test_features, test_label)

    def train_input_fn(self, features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the dataset.

        return dataset

    def eval_input_fn(self, features, labels, batch_size):
        """An input function for evaluation or prediction"""
        features=dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset

    def predict(self,u,d,l,r):
        predict_x = {
            'ABOVE': [u],
            'BELOW': [d],
            'LEFT': [l],
            'RIGHT': [r]
        }    
        predictions = self.classifier.predict(
            input_fn=lambda:self.eval_input_fn(predict_x,
            labels=None,
            batch_size=self.batch_size))
        for pred_dict in predictions:
            return pred_dict['class_ids'][0]

    def train(self):
        (train_x, train_y), (test_x, test_y) = self.load_data()

        self.feature_columns = []
        for key in train_x.keys():
                self.feature_columns.append(tf.feature_column.numeric_column(key=key))

        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=self.feature_columns,
            hidden_units=[10, 10, 10, 10],
            n_classes=256)

        self.classifier.train(
            input_fn=lambda:self.train_input_fn(train_x, train_y,
                                                     self.batch_size),
            steps=self.train_steps)

        # Evaluate the model.
        eval_result = self.classifier.evaluate(
            input_fn=lambda:self.eval_input_fn(test_x, test_y,
                                                    self.batch_size))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

        '''
        # Generate predictions from the model
        expected = [[255,255,255],[255,255,255],[255,255,255]]
        predict_x = {
            'ABOVE': [[255,255,255],[255,255,255],[255,255,255]],
            'BELOW': [[255,255,255],[255,255,255],[255,255,255]],
            'LEFT': [[255,255,255],[255,255,255],[255,255,255]],
            'RIGHT': [[255,255,255],[255,255,255],[255,255,255]]
        }'''


        # Generate predictions from the model
        expected = [255,100,25]
        predict_x = {
            'ABOVE': [255,100,25],
            'BELOW': [255,100,25],
            'LEFT': [255,100,25],
            'RIGHT': [255,100,25]
        }           # PREDICT WHOLE LOT!

        predictions = self.classifier.predict(
            input_fn=lambda:self.eval_input_fn(predict_x,
                                                    labels=None,
                                                    batch_size=self.batch_size))

        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        for pred_dict, expec in zip(predictions, expected):
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]

            print(template.format(class_id,
                                  100 * probability, expec))
    def fill(self, imgdata):
        self.imgdata = imgdata.copy()

        i = 0
        for x in range(0, len(self.imgdata)):       # for each row
            row = self.imgdata[x]
            print('\r\r%s%% done' % round((x/len(self.imgdata)) * 100, 0), end='')
            for y in range(i, len(row), 2):
                #row[y] = self.fillpixel(x,y)
                self.fillpixel(x,y)
            if i == 0:
                i = 1
            else:
                i = 0
        return self.imgdata

    def fillpixel(self, row, col):
        above,below,left,right = self.get_surrounding(row, col)
        
        if None not in [above,below,left,right]:
            r = self.predict(self.imgdata[above[0]][above[1]][0],self.imgdata[below[0]][below[1]][0],self.imgdata[left[0]][left[1]][0],self.imgdata[right[0]][right[1]][0])
            g = self.predict(self.imgdata[above[0]][above[1]][1],self.imgdata[below[0]][below[1]][1],self.imgdata[left[0]][left[1]][1],self.imgdata[right[0]][right[1]][1])
            b = self.predict(self.imgdata[above[0]][above[1]][2],self.imgdata[below[0]][below[1]][2],self.imgdata[left[0]][left[1]][2],self.imgdata[right[0]][right[1]][2])

            self.imgdata[row,col] =  (int(r),int(g),int(b),255)


    def run(self):
        self.generate_data_csv()
        self.save_to_csv()
        self.train()