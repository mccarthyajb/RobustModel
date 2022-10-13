"""More Robust Random Forest classifier."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import SaliencyMapMethod

from art.estimators.classification import KerasClassifier
from art.estimators.classification import SklearnClassifier
from art.estimators.classification import EnsembleClassifier


import tensorflow as tf
from tensorflow import keras
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam


from dictdiffer import diff
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ..robustmodel import RobustModel
from .robustdecisiontreeclassifier import decision_trees_are_equal



    

class RobustRandomForestClassifier(RobustModel, RandomForestClassifier):
    """Privacy protected Random Forest classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        RobustModel.__init__(self)
        RandomForestClassifier.__init__(self, **kwargs)
        self.model_type: str = "RandomForestClassifier"
        super().preliminary_check(apply_constraints=True, verbose=True)
        self.ignore_items = [
            "model_save_file",
            "ignore_items",
            "base_estimator_",
        ]
        self.examine_seperately_items = ["base_estimator", "estimators_"]

    def threshold_slider_moved(self, threshold):
        #print('wobble ' +str( threshold))
        import numpy as np
        from sklearn import datasets
        
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        # Split features and target into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

        self.fit(X_train,y_train)

        # Make predictions for the test set
        #print(X_test)
        y_pred_test = self.predict(X_test)
        # View accuracy score
        orig_acc = accuracy_score(y_test, y_pred_test)

        print(orig_acc)

        epochs=16
        all_features_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=[X_train.shape[1]]),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(y))),
            tf.keras.layers.Activation(tf.nn.softmax)
        ])

        #print(model.summary())

        all_features_model.compile(optimizer='adam',
            #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            loss= 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]
        
        history = all_features_model.fit(
            X_train, y_train, epochs=epochs,
            callbacks=callbacks,
            shuffle=False,
            validation_data=(X_test, y_test),
            verbose=1
        )

        all_features_model.fit(X_train, y_train)


        #art classifier

        art_classifier = KerasClassifier(model=all_features_model, use_logits=False)

        theta=0.50
        gamma=0.50
        attack = SaliencyMapMethod(classifier=art_classifier, theta=theta, gamma=gamma, batch_size=1,verbose=True) # Theta = Small Perturbation , Gamma = 10% of features
        print("Starting to Generate untargeted JSMA")
        #x_test_adv = attack.generate(x=X_test, y=y_test)
        x_test_jsma = attack.generate(x=X_test)

        print("Done!")

        print(x_test_jsma)

        jsma_pred_test = self.predict(x_test_jsma)
        # View accuracy score
        print(y_test)
        print(jsma_pred_test)
        #print(X_test)
        jsma_acc = accuracy_score(y_test, jsma_pred_test)

        print("Reference JSMA Accuracy:" + str(jsma_acc))
        print("Accuracy Difference:" + str(jsma_acc - orig_acc))
        
        reference_score = self.check_features(X,y,threshold)
        reference_acc = orig_acc
        reference_jsma_acc = jsma_acc
        reference_num_features = X_test.shape[1]

        #----------------------------------------------
        #-------- Now reduce features by threshold ----
        #----------------------------------------------

        X, y = self.drop_features(X,y,threshold)

        # Split features and target into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
        #print(X_train.shape)

        self.fit(X_train,y_train)

        # Make predictions for the test set
        y_pred_test = self.predict(X_test)
        # View accuracy score
        #print(y_test)
        #print(y_pred_test)
        #print(X_test)
        new_acc = accuracy_score(y_test, y_pred_test)

        print(new_acc)

        new_score = self.check_features(X,y,threshold)
        #self.plot_features_range(X,y)

        print("Original Accuracy: " + str(orig_acc))
        print("New Accuracy     : " + str(new_acc))
        
        print("Original Feature Score:" + str(reference_score))
        print("New Feature Score     :" + str(new_score))
        new_num_features = X_test.shape[1]
              

        epochs=16
    
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=[X_train.shape[1]]),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(y))),
            tf.keras.layers.Activation(tf.nn.softmax)
        ])

        #print(model.summary())

        model.compile(optimizer='adam',
            #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            loss= 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]
        
        history = model.fit(
            X_train, y_train, epochs=epochs,
            callbacks=callbacks,
            shuffle=False,
            validation_data=(X_test, y_test),
            verbose=1
        )

        model.fit(X_train,y_train)
        
        print("There are " +str(len(X_train)) +" lines in x_train")

        
        #art classifier

        art_classifier = KerasClassifier(model=model, use_logits=False)

        theta=0.50
        gamma=0.50
        attack = SaliencyMapMethod(classifier=art_classifier, theta=theta, gamma=gamma, batch_size=1,verbose=True) # Theta = Small Perturbation , Gamma = 10% of features
        print("Starting to Generate untargeted JSMA")
        #x_test_adv = attack.generate(x=X_test, y=y_test)
        x_test_jsma = attack.generate(x=X_test)

        print("Done!")

        print(x_test_jsma)

        jsma_pred_test = self.predict(x_test_jsma)
        # View accuracy score
        print(y_test)
        print(jsma_pred_test)
        #print(X_test)
        jsma_acc = accuracy_score(y_test, jsma_pred_test)
        new_jsma_acc = accuracy_score(y_test, jsma_pred_test)

        print("Reference JSMA Acc:" + str(reference_jsma_acc))
        print("JSMA Accuracy:" + str(new_jsma_acc))
        print("JSMA Difference:" + str(new_jsma_acc - reference_jsma_acc))
        
        self.dash_features(reference_score,new_score,
                           reference_acc, new_acc,
                           reference_jsma_acc, new_jsma_acc,
                           reference_num_features, new_num_features)
        
    def plot_features_range(self, x: np.ndarray, y: np.ndarray) -> float:
        features =[]
        min_values = []
        max_values = []
        mid_values = []
        range_values =[]
        
        num_features = x.shape[1]
        min_y_limit=0
        max_y_limit=0
        for feature in range(num_features):
            #print(f"feature {feature} min {np.min(x[:,feature])}, max {np.max(x[:,feature])}")
            features.append(feature)
            min_values.append(np.min(x[:,feature]))
            max_values.append(np.max(x[:,feature]))
            mid_values.append((np.max(x[:,feature]) + np.min(x[:,feature])) /2)
            range_values.append(np.max(x[:,feature]) - np.min(x[:,feature]))
            if np.max(x[:,feature]) > max_y_limit:
                max_y_limit = round(np.max(x[:,feature]) +1) 
            else:
                pass
        y1 = min_values
        y2 = max_values
        y3 = mid_values
        plt.figure(figsize=(20,5),dpi=300)
        p1 = plt.plot(features, y1, linewidth=2)
        p2 = plt.plot(features, y2, linewidth=2)
        p3 = plt.plot(features, y3, linewidth=2, linestyle="--")
        plt.xticks(range(0,len(y1)), rotation=90, fontsize=15)
        plt.ylim(min_y_limit,max_y_limit)
        plt.show()
                   
    def dashboard(self):
        #        threshold_slider = widgets.FloatSlider()
        #display(threshold_slider)

        
        w = interact_manual(self.threshold_slider_moved, threshold=widgets.FloatSlider(min=0.0,max=1.0,step=.01,value=0.0))
        display(w)


        
    def dash_features(self,reference_score, feature_score,
                      reference_acc, new_acc,
                      reference_jsma_acc, new_jsma_acc,
                      reference_num_features, new_num_features):

        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = feature_score,
            mode = "gauge+number+delta",
            title = {'text': "Feature Score"},
            delta = {'reference': reference_score},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "gray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': reference_score}}))
        fig.show()

        fig = go.Figure()

        fig.add_trace(go.Indicator(
            #domain = {'x': [0, 1], 'y': [0, 1]},
            value = feature_score,
            mode = "gauge+number+delta",
            title = {'text': "Feature Score"},
            delta = {'reference': reference_score},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "gray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': reference_score}},
            domain = {'row': 1, 'column': 0 }
        ))

        fig.add_trace(go.Indicator(
            #domain = {'x': [0, 1], 'y': [0, 1]},
            value = new_acc,
            mode = "gauge+number+delta",
            title = {'text': "Accuracy"},
            delta = {'reference': reference_acc},
            gauge = {'axis': {'range': [None, 1.00]},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "gray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': reference_acc}},
            domain = {'row': 1, 'column': 1}
        ))

        fig.add_trace(go.Indicator(
            #domain = {'x': [0, 1], 'y': [0, 1]},
            value = new_jsma_acc,
            mode = "gauge+number+delta",
            title = {'text': "Adversarial Accuracy"},
            delta = {'reference': reference_jsma_acc},
            gauge = {'axis': {'range': [None, 1.00]},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "gray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': reference_jsma_acc}},
            domain = {'row': 1, 'column': 2}
        ))

        fig.add_trace(go.Indicator(
            #domain = {'x': [0, 1], 'y': [0, 1]},
            value = new_num_features,
            mode = "gauge+number+delta",
            title = {'text': "Num Features"},
            delta = {'reference': reference_num_features},
            gauge = {'axis': {'range': [None, reference_num_features]},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "gray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': reference_num_features}},
            domain = {'row': 1, 'column': 3}
        ))

        fig.update_layout(
            grid = {'rows': 2, 'columns': 4, 'pattern': "independent"}
            #template = {'data' : {'indicator': [{
            #    'title': {'text': "Speed"},
            #    'mode' : "number+delta+gauge",
            #    'delta' : {'reference': 90}}]
            #}})
            )
        
        fig.show()

        
    def drop_features(self,x: np.ndarray, y : np.ndarray, threshold: float) -> tuple[np.ndarray,np.nd.array]:
        new_x = x
        features_to_drop = []
        num_features = x.shape[1]
        for feature in range(num_features):
            #print(f"feature {feature} min {np.min(x[:,feature])}, max {np.max(x[:,feature])}")
            
            # get importance
            importance = self.feature_importances_
            # summarize feature importance
            num_important_features = 0
            cumulative_score = 0
            for feature,v in enumerate(importance):
                cumulative_score = cumulative_score + v
                #print('Feature: %0d, Score: %.5f' % (feature,v))
                #print("threshold" + str(threshold))
                #print("v" + str(v))
                if threshold >= v:
                    features_to_drop.append(feature)
                    #print("features to drop")
                    #print(features_to_drop)

            new_x = np.delete(x, [features_to_drop], axis=1)
            #print(new_x.shape)
            if(new_x.shape[1] <= 0):
                print("WARNING: Importance Threshold excludes all features")
                return x, y
            else:
                
                return new_x, y
 
        
        
    def check_features(self, x: np.ndarray, y: np.ndarray, threshold: float) -> float:
        #print("Shape of X is {}".format(x.shape))
        num_features = x.shape[1]
        #print(num_features)
        
        for feature in range(num_features):
            #print(f"feature {feature} min {np.min(x[:,feature])}, max {np.max(x[:,feature])}")
            pass

        # get importance
        importance = self.feature_importances_
        # summarize feature importance
        num_important_features = 0
        cumulative_score = 0
        for i,v in enumerate(importance):
            cumulative_score = cumulative_score + v
            #print('Feature: %0d, Score: %.5f' % (i,v))
            if v > threshold:
                num_important_features = num_important_features +1
            else:
                pass

        feature_score = num_important_features / num_features * 100
        average_score = cumulative_score / num_features * 100

        #print('Feature Score ' + str(feature_score))        
        
        # plot feature importance
        plt.xticks(range(0,len(importance)), rotation=90, fontsize=15)
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()


        #print(average_score)
        return average_score

        
        
        
    def additional_checks(
                self, curr_separate: dict, saved_separate: dict
        ) -> tuple[str, str]:
            """Random Forest-specific checks"""

            # call the super function to deal with any items that are lists
            msg, disclosive = super().additional_checks(curr_separate, saved_separate)
            # now the relevant random-forest specific things
            for item in self.examine_seperately_items:
                if item == "base_estimator":
                    try:
                        the_type = type(self.base_estimator)
                        if not isinstance(self.saved_model["base_estimator_"], the_type):
                            msg += "Warning: model was fitted with different base estimator type.\n"
                            disclosive = True
                    except AttributeError:
                        msg += "Error: model has not been fitted to data.\n"
                        disclosive = True

                elif item == "estimators_":

                    if curr_separate[item] == "Absent" and saved_separate[item] == "Absent":
                        disclosive = True
                        msg += "Error: model has not been fitted to data.\n"
                        
                    elif curr_separate[item] == "Absent":
                        disclosive = True
                        msg += "Error: current version of model has had trees removed after fitting.\n"

                    elif saved_separate[item] == "Absent":
                        disclosive = True
                        msg += "Error: current version of model has had trees manually edited.\n"
                        
                    else:
                        try:
                            num1 = len(curr_separate[item])
                            num2 = len(saved_separate[item])
                            if num1 != num2:
                                msg += (
                                    f"Fitted model has {num2} estimators "
                                    f"but requested version has {num1}.\n"
                                )
                                disclosive = False
                            else:
                                for idx in range(num1):
                                    same, msg2, = decision_trees_are_equal(
                                        curr_separate[item][idx], saved_separate[item][idx]
                                    )
                                    if not same:
                                        disclosive = True
                                        msg += f"Forest base estimators {idx} differ."
                                        msg += msg2
                                        
                        except BaseException as error:
                            msg += (
                                "In SafeRandomForest.additional_checks: "
                                f"Unable to check {item} as an exception occurred: {error}.\n"
                            )
                            same = False

                elif isinstance(curr_separate[item], DecisionTreeClassifier):
                    diffs_list = list(diff(curr_separate[item], saved_separate[item]))
                    if len(diffs_list) > 0:
                        disclosive = True
                        msg += f"structure {item} has {len(diffs_list)} differences: {diffs_list}"
                return msg, disclosive

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Do fit and then store model dict"""
        super().fit(x, y)
        self.k_anonymity = self.get_k_anonymity(x)
        self.saved_model = copy.deepcopy(self.__dict__)

    def get_k_anonymity(self, x: np.ndarray) -> int:
        """calculates the k-anonymity of a random forest model
        as the minimum of the anonymity for each record.
        That is defined as the size of the set of records which
        appear in the same leaf as the record in every tree.
        """

        # dataset must be 2-D
        assert len(x.shape) == 2

        num_records = x.shape[0]
        num_trees = self.n_estimators
        k_anon_val = np.zeros(num_records, dtype=int)

        # ending leaf node by record(row) and tree (column)
        all_leaves = np.zeros((num_records, num_trees), dtype=int)
        for this_tree in range(num_trees):
            this_leaves = self.estimators_[this_tree].apply(x)
            for record in range(num_records):
                all_leaves[record][this_tree] = this_leaves[record]

        for record in range(num_records):
            # start by assuming everything co-occurs
            appears_together = list(range(0, num_records))
            # iterate through trees
            for this_tree in range(num_trees):
                this_leaf = all_leaves[record][this_tree]

                together = copy.copy(appears_together)
                # removing records which go to other leaves
                for other_record in together:
                    if all_leaves[other_record][this_tree] != this_leaf:
                        appears_together.remove(other_record)

            k_anon_val[record] = len(appears_together)
        return k_anon_val.min()
