import utils
import model_module
import tensorflow as tf
import argparse

if __name__ == "__main__":
    # create and parse arguments
    parser = argparse.ArgumentParser(
        description="A python script for performing 2D latent space exploration and image generation with TP-SE interactively.")
    parser.add_argument("--weightPath", type=str,
                        action="store", metavar="PATH",
                        required=True, help="The path to the trained TP-SE weights (2D latent space).")
    parser.add_argument("--xbounds", type=float,
                        action="store", nargs=2,
                        metavar=("X_MIN", "X_MAX"), required=False,
                        default=[-25, 25], help="The min and max values for the x-axis.")
    parser.add_argument("--ybounds", type=float,
                        action="store", nargs=2,
                        metavar=("Y_MIN", "Y_MAX"), required=False,
                        default=[-25, 25], help="The min and max values for the y-axis.")
    parser.add_argument("--nPoints", type=utils.non_negative_int_input,
                        action="store", required=False,
                        default=150, help="The number of points to obtain between the specified min max intervals.")
    parser.add_argument("--targetDirectory", type=str,
                        default=".", required=False,
                        help="The target directory to save the generated figures (if it does not exist,\
                                 it will be created).")
    parser.add_argument("--format", type=str, default="jpg", choices=["jpg", "png", "pdf"],
                        required=False, help="The format to save the generated figures (choices: png, jpg, pdf).")
    parser.add_argument("--zoom", type=utils.non_negative_float_input, default=0.7,
                        required=False, help="The zoom utilized for the decoded images.")
    parser.add_argument("--figSize", type=utils.non_negative_int_input,
                        action="store", nargs=2,
                        metavar=("WIDTH", "HEIGHT"), required=False,
                        default=[12, 8], help="The figure size in inches.")
    args = parser.parse_args()

    # set some more variables
    shape_of_input = [28, 28, 1]
    latent_space_dimensions = 2
    num_classes = 10

    # Build the autoencoder network (encoder and decoder networks) and the Separator network
    autoencoder, class_model = model_module.build_test_model(shape_of_input,
                                                             latent_dims=latent_space_dimensions,
                                                             number_of_classes=num_classes)

    # Instantiate the Two Phase Supervised Encoder
    tpse = model_module.TPSE(autoencoder=autoencoder,
                             class_model=class_model)

    tpse.build((None, shape_of_input[0], shape_of_input[1], shape_of_input[2]))

    # load the specified model weights
    tpse.load_weights(args.weightPath)

    # Get the Decoder model
    decoder_input = tf.keras.Input((2,))
    first = False
    is_first_entry = True
    x = None
    for layer in tpse.autoencoder.layers:
        if first or ("Decoder1" in layer.name):
            if is_first_entry:
                is_first_entry = False
                first = True
                x = layer(decoder_input)
            else:
                x = layer(x)

    decoder = tf.keras.Model(inputs=decoder_input, outputs=x)

    # Get the Separator model
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

    # start the interactive 2D latent space exploration and image generation
    utils.latent_space_exploration_image_generation(classifier,
                                                    decoder,
                                                    xbounds=args.xbounds,
                                                    ybounds=args.ybounds,
                                                    zoom=args.zoom,
                                                    figsize=args.figSize,
                                                    cmap="gray",
                                                    number_of_grid_points=args.nPoints,
                                                    save_directory=args.targetDirectory,
                                                    output_format=args.format
                                                    )
