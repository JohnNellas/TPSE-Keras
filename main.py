import tensorflow as tf
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.svm
import sklearn.naive_bayes
import sklearn.manifold
import sklearn.decomposition
import model_module
import os
import argparse
import utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(mode: str,
         latent_space_dimensions: int,
         epochs: int,
         batch_size: int,
         weight_path: str = None,
         learning_rate: float = 1e-3,
         validation_size: float = 0.1,
         zoom: float = 0.5,
         figsize: tuple = (12, 10),
         experiment_id: str = "out",
         dpi: int = 300,
         seed_value: int = 42,
         number_of_grid_points: int = 150,
         out_format: str = "jpg",
         n_neighbors: int = 3):
    """
    A function for executing the experiment on MNIST dataset utilizing the
    Two Phase Supervised Encoder.
    :param mode: Run the experiment for training (t), evaluation (e) or both (te).
    :param latent_space_dimensions: The dimensionality of the latent space.
    :param epochs: The number of epochs when training.
    :param batch_size: The batch size.
    :param weight_path: The path to the trained TP-SE weights for evaluation mode.
    :param learning_rate: The learning rate.
    :param validation_size: The portion of the training set used for validation when training.
    :param zoom: The zoom of the displayed images in the explainability 2D Scatter Plot.
    :param figsize: The figure size as a tuple (width, height) in inches.
    :param experiment_id: The identifier of the experiment.
    :param dpi: The employed dpi when saving the generated figures.
    :param seed_value: The seed value.
    :param number_of_grid_points: The number of points to sample per dimension in the 2D scatter plot of explainability.
    :param out_format: The format to save the generated figures (choices: png, jpg, pdf).
    :param n_neighbors: The number of nearest neighbors utilized for the kNN classifier when evaluating.
    :return: None
    """

    # load the train and test set of the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # set some dataset specific settings
    shape_of_input = [28, 28, 1]
    num_classes = 10
    cmap = "gray"

    # create a validation set that is equal to the validation_size % of the train set
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train,
                                                                              y_train,
                                                                              test_size=validation_size,
                                                                              shuffle=True,
                                                                              stratify=y_train,
                                                                              random_state=seed_value)

    # normalize the images
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    # create the Channels Last format
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # build the autoencoder network (encoder and decoder networks) and the Separator network
    autoencoder, class_model = model_module.build_test_model(shape_of_input,
                                                             latent_dims=latent_space_dimensions,
                                                             number_of_classes=num_classes)

    # instantiate the Two Phase Supervised Encoder
    tpse = model_module.TPSE(autoencoder=autoencoder,
                             class_model=class_model)

    tpse.build((None, x_train.shape[1], x_train.shape[2], x_train.shape[3]))

    # see a summary of the model
    tpse.summary()

    # =============================================================================
    # FILE STRUCTURE CREATION
    # =============================================================================
    if (mode == "t") or (mode == "te"):
        checkpoint_parent_path = os.path.join(".", f"checkpoints_tpse{experiment_id}")
        if not os.path.isdir(checkpoint_parent_path):
            os.mkdir(checkpoint_parent_path)

        checkpoint_path = os.path.join(checkpoint_parent_path,
                                       f"checkpoints_mnist_{latent_space_dimensions}_dimensions")
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)

    if (mode == "e") or (mode == "te"):
        classification_results_parent_path = os.path.join(".", f"classification_results_tpse{experiment_id}")
        if not os.path.isdir(classification_results_parent_path):
            os.mkdir(classification_results_parent_path)

        classification_results_middle_path = os.path.join(classification_results_parent_path, "classification_results")
        if not os.path.isdir(classification_results_middle_path):
            os.mkdir(classification_results_middle_path)

        visualization_results_middle_path = os.path.join(classification_results_parent_path, "visualization_results")
        if not os.path.isdir(visualization_results_middle_path):
            os.mkdir(visualization_results_middle_path)

        classification_results_path = os.path.join(classification_results_middle_path,
                                                   f"classification_results_mnist_{latent_space_dimensions}_dimensions")
        if not os.path.isdir(classification_results_path):
            os.mkdir(classification_results_path)

        visualization_results_path = os.path.join(visualization_results_middle_path,
                                                  f"visualization_results_mnist_{latent_space_dimensions}_dimensions")
        if not os.path.isdir(visualization_results_path):
            os.mkdir(visualization_results_path)

    # ===============================================
    # Training Settings
    # ===============================================

    if (mode == "t") or (mode == "te"):
        # set the optimizer for both phases
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # set the loss functions for the unsupervised and supervised phases
        loss_function_autoencoder = tf.keras.losses.MeanSquaredError()
        loss_function_classifier = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # set the metrics for the unsupervised and supervised phases
        ae_metric = tf.keras.metrics.MeanSquaredError(name="ae_loss")
        class_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True,
                                                                      name="class_loss")

        # compile the model
        tpse.compile(class_optimizer=optimizer,
                     ae_optimizer=optimizer,
                     class_loss_fn=loss_function_classifier,
                     ae_loss_fn=loss_function_autoencoder,
                     class_metric=class_metric,
                     ae_metric=ae_metric)

        # define the callbacks
        callbacks = [tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, "best_weights.hdf5"),
                                                        monitor="val_accuracy",
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        mode="max",
                                                        verbose=1
                                                        ),
                     tf.keras.callbacks.LearningRateScheduler(
                         model_module.wrapper_lr_schedule(number_of_epochs=epochs),
                         verbose=0)
                     ]

        # start the training process
        tpse.fit(x_train,
                 y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(x_val, y_val),
                 shuffle=True,
                 callbacks=callbacks,
                 verbose=1)

    if (mode == "e") or (mode == "te"):

        # =============================================================================
        # Weight Loading
        # =============================================================================
        print("Loading weights...")

        if mode == "e":
            # if evaluation mode: get the user specified weights
            weights_path = weight_path
        else:
            # if train and evaluation mode: get the best weights from training
            weights_path = os.path.join(checkpoint_path, f"best_weights.hdf5")

        # load the weights
        tpse.load_weights(weights_path)

        # isolate the Encoder network
        encoder = tf.keras.Model(inputs=tpse.autoencoder.input,
                                 outputs=tpse.autoencoder.get_layer("LatentLayer").output)

        # get the Separator model
        class_input = tf.keras.Input((latent_space_dimensions,))
        first = False
        is_first_entry = True
        x = None
        for layer in tpse.class_model.layers:
            if first or ("Class1" in layer.name):
                if is_first_entry:
                    is_first_entry = False
                    first = True
                    x = layer(class_input)
                else:
                    x = layer(x)

        classifier = tf.keras.Model(inputs=class_input, outputs=x)

        # get the embedding of the train and test sets
        lr_train = encoder.predict(x_train)
        lr_test = encoder.predict(x_test)

        # =============================================================================
        # CLASSIFICATION - Separator Network
        # =============================================================================

        # evaluate the performance of the separator network on the test set and save the results
        predictions = tpse.class_model.predict(x_test).argmax(axis=1)

        filename = os.path.join(classification_results_path, f"mnist_classification_{latent_space_dimensions}.txt")
        cls_rep_str, _, analytic_string = utils.evaluate_method(y_test,
                                                                predictions,
                                                                save_flag=True,
                                                                filename=filename)

        # print the classification report
        print(cls_rep_str)
        print(analytic_string)

        # ============================================================================
        # Classification Methods on Latent Space
        # ============================================================================

        print(
            "------------------------------ Improving Traditional Methods -----------------------------------------")

        print("---------------------------- k-Nearest Neighbors ---------------------")
        # apply knn on the derived test latent representations, evaluate and save the results
        filename = os.path.join(classification_results_path, f"mnist_KNN_{n_neighbors}_{num_classes}_classes.txt")
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        cls_rep_str, _, analytic_string = utils.evaluate_with_classifier(classifier_object=knn,
                                                                         x_tr=lr_train,
                                                                         y_tr=y_train,
                                                                         x_te=lr_test,
                                                                         y_te=y_test,
                                                                         save_flag=True,
                                                                         filename=filename
                                                                         )
        # print the classification report
        print(cls_rep_str)
        print(analytic_string)

        print("---------------------------- SVM with RBF ---------------------")
        # apply svm with RBF kernel on the derived test latent representations, evaluate and save the results
        filename = os.path.join(classification_results_path, f"mnist_SVM_{num_classes}_classes.txt")
        svms = sklearn.svm.SVC(random_state=seed_value)
        cls_rep_str, _, analytic_string = utils.evaluate_with_classifier(classifier_object=svms,
                                                                         x_tr=lr_train,
                                                                         y_tr=y_train,
                                                                         x_te=lr_test,
                                                                         y_te=y_test,
                                                                         save_flag=True,
                                                                         filename=filename
                                                                         )

        # print the classification report
        print(cls_rep_str)
        print(analytic_string)

        print("---------------------------- Naive Bayes ---------------------")
        # apply the Naive Bayes Classifier on the derived test latent representations, evaluate and save the results
        filename = os.path.join(classification_results_path, f"mnist_GNB_{num_classes}_classes.txt")
        gnb = sklearn.naive_bayes.GaussianNB()
        cls_rep_str, _, analytic_string = utils.evaluate_with_classifier(classifier_object=gnb,
                                                                         x_tr=lr_train,
                                                                         y_tr=y_train,
                                                                         x_te=lr_test,
                                                                         y_te=y_test,
                                                                         save_flag=True,
                                                                         filename=filename
                                                                         )
        # print the classification report
        print(cls_rep_str)
        print(analytic_string)

        # =============================================================================
        #     VISUALIZATIONS - TEST EMBEDDING
        # =============================================================================

        # =============================================================================
        #     CREATE A 2D VISUALIZATION USING PCA AND TSNE
        # =============================================================================
        if latent_space_dimensions > 3:

            # =============================================================================
            #     TSNE VISUALIZATION
            # =============================================================================
            print("Producing the test embedding visualization using TSNE...")
            embs_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(lr_test)

            emb_df = pd.DataFrame(embs_tsne, columns=["E1", "E2"])
            emb_df["y"] = y_test
            emb_df["y"] = emb_df["y"].astype('category')

            fig, ax = plt.subplots(1, 1, figsize=figsize)

            sns.scatterplot(ax=ax, x="E1", y="E2", data=emb_df, hue="y")
            ax.axis("off")
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # save the final figure
            plt.savefig(
                os.path.join(visualization_results_path,
                             f"embeddings_tsne_mnist_{latent_space_dimensions}.{out_format}"),
                dpi=dpi,
                bbox_inches='tight')
            plt.close()

            # =============================================================================
            #     PCA VISUALIZATION
            # =============================================================================
            print("Producing the test embedding visualization using PCA...")
            embs_pca = sklearn.decomposition.PCA(n_components=2).fit_transform(lr_test)

            emb_df = pd.DataFrame(embs_pca, columns=["E1", "E2"])
            emb_df["y"] = y_test
            emb_df["y"] = emb_df["y"].astype('category')

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            sns.scatterplot(ax=ax, x="E1", y="E2", data=emb_df, hue="y")
            ax.axis("off")
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # save the final figure
            plt.savefig(
                os.path.join(visualization_results_path,
                             f"embeddings_pca_mnist_{latent_space_dimensions}.{out_format}"),
                dpi=dpi,
                bbox_inches='tight')
            plt.close()

        elif latent_space_dimensions == 3:
            # =============================================================================
            #     VISUALIZATION OF RAW 3D EMBEDDING
            # =============================================================================
            print("Visualizing the 3D raw test embedding...")
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            for uq_class in np.unique(y_test):
                idx = np.where(y_test == uq_class)[0]
                ax.scatter(lr_test[idx, 0], lr_test[idx, 1], lr_test[idx, 2], label=uq_class)

            ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # save the final figure
            plt.savefig(os.path.join(visualization_results_path,
                                     f"embeddings_{latent_space_dimensions}D_mnist.{out_format}"),
                        bbox_inches='tight',
                        dpi=dpi)
            plt.close()

        else:
            # =============================================================================
            #     VISUALIZATION OF RAW 2D EMBEDDING
            # =============================================================================
            print("Visualizing the 2D raw test embedding...")
            emb_df = pd.DataFrame(lr_test, columns=["E1", "E2"])
            emb_df["y"] = y_test
            emb_df["y"] = emb_df["y"].astype('category')

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            sns.scatterplot(ax=ax, x="E1", y="E2", data=emb_df, hue="y")
            ax.axis("off")
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # save the final figure
            plt.savefig(
                os.path.join(visualization_results_path,
                             f"embeddings_mnist_{latent_space_dimensions}_grid.{out_format}"),
                dpi=dpi,
                bbox_inches='tight')
            plt.close()

            print("Producing the 2D test embedding Scatter Plot of Explainability...")

            # additional 2D test embedding Scatter Plot of explainabiliy:
            # Latent Space, decision boundaries and regions of the separator network
            # along with replacements of some test embeddings per class with the original test image.

            utils.decision_boundary_on_latent_space(classifier,
                                                    embeddings=lr_test,
                                                    ground_truth=y_test,
                                                    zoom=zoom,
                                                    figsize=figsize,
                                                    save_flag=True,
                                                    save_path=os.path.join(visualization_results_path,
                                                                           f"decision_boundary_mnist.{out_format}"),
                                                    number_of_grid_points=number_of_grid_points,
                                                    display_images_flag=True,
                                                    images_sample_pool=x_test,
                                                    cmap=cmap,
                                                    dpi=dpi,
                                                    )


if __name__ == "__main__":
    # create and parse arguments
    parser = argparse.ArgumentParser(
        description="A python script for executing the experiment on MNIST dataset \
         utilizing the Two Phase Supervised Encoder.")
    parser.add_argument("--mode", type=str,
                        action="store", metavar="MODE",
                        choices=["t", "e", "te"],
                        required=True,
                        help="Run the experiment for training (t), evaluation (e) or both (te).")
    parser.add_argument("--latentSpaceDims", type=utils.non_negative_int_input,
                        action="store", metavar="LATENT_SPACE_DIMS",
                        required=True, help="The dimensionality of the latent space.")
    parser.add_argument("--weightPath", type=str,
                        action="store", metavar="PATH", required=False,
                        default=None, help="The path to the trained TP-SE weights for evaluation mode.")
    parser.add_argument("--epochs", type=utils.non_negative_int_input,
                        action="store", metavar="EPOCHS", required=False,
                        default=200, help="The number of epochs when training.")
    parser.add_argument("--batch_size", type=utils.non_negative_int_input,
                        action="store", metavar="BATCH_SIZE", required=False,
                        default=128, help="The batch size.")
    parser.add_argument("--lr", type=utils.non_negative_float_input,
                        action="store", metavar="LR", required=False,
                        default=1e-3, help="The learning rate.")
    parser.add_argument("--valSize", type=utils.non_negative_float_input,
                        default=0.1, metavar="VAL_SIZE",
                        required=False, help="The portion of the training set used for validation when training.")
    parser.add_argument("--nPoints", type=utils.non_negative_int_input,
                        action="store", required=False,
                        default=150, metavar="N_POINTS",
                        help="The number of points to sample per dimension in the 2D scatter plot of explainability.")
    parser.add_argument("--nNeighbors", type=utils.non_negative_int_input,
                        action="store", required=False,
                        default=3, metavar="N_NEIGHBORS",
                        help="The number of nearest neighbors utilized for the kNN classifier when evaluating.")
    parser.add_argument("--dpi", type=utils.non_negative_int_input,
                        action="store", required=False, metavar="DPI",
                        default=300, help="The employed dpi when saving the generated figures.")
    parser.add_argument("--format", type=str,
                        default="jpg", choices=["jpg", "png", "pdf"],
                        required=False, help="The format to save the generated figures (choices: png, jpg, pdf).")
    parser.add_argument("--experimentID", type=str,
                        default="out", required=False,
                        metavar="EXPERIMENT_ID", help="The identifier of the experiment.")
    parser.add_argument("--zoom", type=utils.non_negative_float_input,
                        default=0.5, required=False, metavar="ZOOM",
                        help="The zoom of the displayed images in the explainability 2D Scatter Plot.")
    parser.add_argument("--seedVal", type=utils.non_negative_int_input,
                        default=42, required=False,
                        metavar="SEED_VAL", help="The seed value.")
    parser.add_argument("--figSize", type=utils.non_negative_float_input,
                        action="store", nargs=2,
                        metavar=("WIDTH", "HEIGHT"), required=False,
                        default=[12, 10], help="The figure size in inches.")
    args = parser.parse_args()

    # run the experiment with the provided and default argument values
    main(mode=args.mode,
         latent_space_dimensions=args.latentSpaceDims,
         epochs=args.epochs,
         batch_size=args.batch_size,
         weight_path=args.weightPath,
         learning_rate=args.lr,
         validation_size=args.valSize,
         zoom=args.zoom,
         figsize=args.figSize,
         experiment_id="_" + args.experimentID,
         dpi=args.dpi,
         seed_value=args.seedVal,
         number_of_grid_points=args.nPoints,
         out_format=args.format,
         n_neighbors=args.nNeighbors)
