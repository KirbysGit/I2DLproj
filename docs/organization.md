i2dlProj /

    / checkpoints 
        / best_model.pth
            Holds our best model desirably.

        / checkpoint_epoch_x.pth
            Holds checkpoints per epoch ran.

    / datasets

        / annotations
            . annotations_test.csv
                Holds annotations for test images.

            . annotations_train.csv
                Holds annotations for training images.

            . annotations_val.csv
                Holds annotations for validation data.

        / images 
            / tests
            / train
            / val

        . SKU-110K.yaml

    / debug_output
    
    / docs
        . architecture.md
            Details architecture of our model.
        . dev.md
            Development details for our project.

    / src

        / config
            . config.yaml - Master Settings File

                Defines global settings for how the entire training pipeline behaves.
                
                Contains main config settings such as :
                    - Dataset Paths.
                    - Development/Test Mode Settings.
                    - Data Preprocessing Parameters.
                    - Model Architecture Configuration.
                    - Training Parameters.
                    - Evaluation Thresholds.
                    - Logging & Checkpointing.
                    - Error Handling Rules.

            . test_config.yaml - Testing Mode Config

                Used for quick debugging & small-scale training runs.

                Contains reduced settings for : 
                    - Lightweight Model Parameters.
                    - Debug-Friendly Training Settings.
                    - Testing-Specific Controls.
                    - Visualization Options.
                    - Custom Save Directories for Test Outputs.

            . train_config.yaml - Full Training Config

                Used during full-scale training runs on the complete dataset.

                Contains core settings for :
                    - Model Architecture.
                    - Dataset Paths & Images Resizing.
                    - Training Hyperparameters.
                    - Checkpointing & Logging.
                    - Evaluation Thresholds.
                    - Optimizer & Scheduler Options.
                    - Visualization During Training.
                    - WandB & Verbose Logging Support. ? 

        / data

            . dataset.py - Dataset Loader for SKU-110K (CV2 + Torch)

                Defines how to load, preprocess, and return each training/validations/test sample.
                
                Simple Breakdown :
                - 1. Load CSV annotations and link them to each image file.
                - 2. Per Image, Open it w/ OpenCV, Get Bounding Boxes and Labels.
                - 3. Return a ready-to-use sample with image tensor + bounding box info.
                - 4. During training, batches are formed using collate_fn to handle different box counts per image.

                Handles :
                - Loads Annotations & Image Paths.
                - Parses & Validates Bounding Boxes.
                - Applies Image Transformations (Resizing, Normalization, Augmentation).
                - Prepares data for Training or Validation.

                Functions :
                - _init_ -> Loads annotation CSV, creates image path mappings.
                - _len_ -> Returns number of unique images.
                - get_image_annotations -> Extracts bounding boxes & labels for a given image.
                - resize_with_aspect_ratio -> Resizes & Pads the image while adjusting boxes accordingly.
                - _getitem_ -> Loads and returns a single sample (image, boxes, labels, image ID).
                - collate_fn -> Batches samples into a consistent format.

                Outputs : 

                - Dictionary :
                    - image -> Tensor[3, H, W]
                    - boxes -> Tensor[N, 4]
                    - labes -> Tensor[N]
                    - image_id -> Image Filename String.

        / models 

            . _init_.py - Makes a Directory a Python Package.
                
            . anchor_generator.py - Anchor Box Generator for Detection.

                - Generates anchor boxes at different scales and aspect ratios across FPN levels. Used for matching predicted boxes to ground truth during training.

                Simple Breakdown :
                - 1. Build anchor templates (called base anchors) for different sizes and shapes.
                - 2. Spread those templates across every pixel location of each feature map.
                - 3. Get a giant list of anchor boxes coverign the full image at different resolutions.
                - 4. These anchors are used to detect objects and compare with ground truth boxes during training.

                Handles : 
                - Precomputes base anchors at multiple scales/aspect ratios.
                - Maps base anchors across feature maps to generate full grid of anchors.
                - Supports single-level and multi-level generation.
                - Vaidates anchor dimensions and prints warning for invalid ones.

                Functions :
                - _init_ -> Sets base sizes, ratios, and scales. Precomputes base anchor shapes.
                - _generate_base_anchors -> Builds a set of anchors centered at (0, 0) per pyramid level.
                - generate_anchors -> Projects anchors across spatial locations of each feature map level.
                - generate_anchors_for_level -> Does same as above but for single feature map.
                - _repr_ -> Pretty-print Config when Logging or Debugging.

                Output :
                - List of Tensors : Each has shape [H x W x A, 4] for a feature map level.

            . backbone.py - ResNet50 Backbone for Feature Extraction.

                - Defines the core feature extractor using a pretrained ResNet50 model. Extracts multiscale feature maps that feed into the FPN and detections heads.

                Simple Breakdown :
                - 1. Input an image.
                - 2. Send it through early convs (layer0).
                - 3. Pass through ResNet blocks (layer1 - layer4).
                - 4. Return the feature maps at each stage.
                - 5. These maps help the model detect objects at different sizes.

                Purpose :
                - Acts as the encoder or backbone of the detection pipeline.
                - Outputs intermediate feature maps at multiple spatial resolutions.
                - Supports pretrained weights for transfer learning.

                Handles :
                - Loads ResNet50 from torchvision.models.
                - Removes classification head (fc layer) and retains convolutional layers.
                - Returns freatures from different RestNet for FPN-style multiscale detection.

                Structure :
                - layer0 : Initial Conv -> BN -> ReLU -> MaxPool (pre-layer1).
                - layer1 : First residual block (256).
                - layer2 : Second residual block (512).
                - layer3 : Third residual block (1024).
                - layer4 : Final residual block (2048).

                Functions :
                - _init_ -> Loads and sets up teh modified ResNet50 layers.
                - forward(x) -> Passes input through layers, Collects and return features at each stage as a dictionary.

                Outputs :
                - features -> Dictionary of tensors : layer0, layer1, layer3, layer4
                - Each tensor : Shape (B, C, H, W) where spatial dimensions decrease with depth.

            . detection_head.py - Detection Head for Anchor-Based Object Detection

                - Defines core detection module responsible for predicting object class scores and bounding box regressions on feature maps and generated anchors. Also, handles anchor matching and loss computation. 

                Simple Breakdown :

                - 1. Make Predictions.
                - 2. Generate Anchors.
                - 3. Matches Anchors to Real Boxes.
                - 4. Calculates Loss.
                
                Dependencies :
                - src / utils / box_ops -> For computing IoUs.
                - src / model / anchor_generator -> To generate anchors.

                Purpose :
                - Predicts class probabilities and bounding box offsets for each anchor.
                - Matches anchors to ground truth using IoU.
                - Computes classification and box regression losses.
                - Converts delats <-> boxes via BoxCoder.

                Handles :
                - Generates anchors using AnchorGenerator.
                - Define Prediction heads.
                - Matches anchors to ground truth targets with IoU filtering.
                - Applies delates to anchors to generate predicted boxes.
                - Computes classification and regression losses.
                - Coverts between coordinate formats using BoxCoder.

                Structure :
                - BoxCoder - Utility to convert beween box coordinates and deltas.
                - Detection Head - Main module for making predictions and computing losses.

                Functions :
                - _init_ -> Builds detection head, anchor generator, and loss functions.
                - _initialize_weights -> Appling Kaimin init to all conv layers.
                - forward_single -> Run shared convs on one feature maps and returns class logits + box deltas.
                - match_anchors_to_targets -> Matches anchors to GT boxes using IoU and assigns labels.
                - forward -> Full forward pass : generates anchors, run head, applies deltas to anchors.
                - cls_loss -> Computes classification loss by matching anchors to labels.
                - box_loss -> Computes regression loss between predicted and GT boxes (only for positives).
                - assign_targets -> Assigns each anchor the label of its highest-IoU GT match.

                Outputs :

                - Forward Pass :
                    {
                        'cls_scores' : List[Tensor[B, A*C, H, W]],
                        'bbox_preds' : List[Tensor[B, A, 4, H, W]], 
                        'anchors' : List[Tesnor[H*W*A, 4]]
                    }

                - Losses :
                    - cls_loss : scalar classification loss.
                    - box_loss : scalar regression loss.

            . detector.py - Complete Object Detection Model.   
                
                - Combines the backbone, FPN, and detection head into a full object detection pipeline.

                Simple Breakdown : 
                - 1. Input an Image Batch.
                - 2. Extract multiscale features using ResNet Backbone.
                - 3. Refine featuers using the Feature Pyramid Network (FPN).
                - 4. Predicts class scores and box deltas with the Detection Head.
                - 5. Post-process outputs with confidence filtering and NMS.

                Purpose :
                - Acts as the full detection pipeline from input to output predictions.
                - Integrates all submodules: ResNet, FPN, and Detection Head.

                Dependencies :
                - backbone.py - 
                - fpn.py - 
                - detection_head.py - 

                Handles : 
                - Loads and connects all modules.
                - Applies class score thresholds and box regression deltas.
                - Applies filtering and Non-Maximum Suppresion (NMS).
                - Supports loading from checkpoints and configurable behavior via config.

                Structure :
                - ObjectDetector - Full Detection Model.
                - DetectionModel - Alternate or legacy structure for modular predictions.
                - build_detector - Constructs model from a config directory.

                Functions :
                - _init_ -> Initializes detector with Backbone, FPN, and head, along with thresholds and scaling.
                - from_checkpoint -> Loads a saved model with fallback strategies.
                - forward -> Full detection pass : images -> feature maps -> predictions -> NMS -> final outputs.
                - build_detector(config) -> Factory function to create and connect modules based on config.
                - apply_deltas_to_anchors -> Converts predicted deltas + anchors to actual box coordinates.

                Outputs :
                - During Inference : dict with boxes, scores, labels, all_scores.
                - During Training : dict with pred_boxes, pred_scores, gt_boxes, gt_labels.

            . fpn.py - Feature Pyramid Network for Multiscale Detection.
                
                - Builds multi-scale feature maps by combining low and high level backbone features.

                Simple Breakdown :
                - 1. Take features from the backbone (C2 -> C5).
                - 2. Apply 1 x 1 convs to normalize channels (lateral connections).
                - 3. Upsample and add top-down features.
                - 4. Apply 3 x 3 convs to clean fused outputs.
                - 5. Return features as P2 to P5 (for detection at different scales).

                Purpose :
                - Enhance low-res features with semantic context from high-res layers.
                - Allows the detector to handle objects of different sizes.
                - Keeps consistent numbers of channels across all FPN levels.

                Handles :
                - Lateral 1 x 1 convolutions to align channels.
                - Top-down upsampling and addition of features.
                - Output 3 x 3 convolutions to refine final feature maps.
                - Reverses order (top-down) for FPN construction.

                Structure :
                - lateral_convs : 1 x 1 convs to project backbone features to common dimension.
                - output_convs : 3 x 3 convs to finalize P2 - P5 features.
                - upsample_add -> Utility for nearest-neighbor upsampling + element-wise addition.

                Functions :
                - _init_ -> Initializes FPN with lateral and output conv layers, set up weights.
                - _upsample_add(x,y) -> Upsamples x to match y's size and adds them together.
                - forward(features) -> Builds FPN outputs (p2 to p5) using lateral and top-down fusion.

                Outputs :
                - features -> Dictionary {'p2' : ... , 'p3' : ... , 'p4' : ... , 'p5' : ... }
                - Each tensor shape (B, out_channels, H, W) with decreasing spatial resolution.

            . losses.py - Loss Functions for Object Detection.

                - Implements key loss components : Focal Loss, IoU Loss, Detection Loss, IoU-Weighted Loss.

                Simple Breakdown :
                - FocalLoss - Handles class imbalances by focusing on hard examples.
                - IoULoss - Directly optimizes overlap between predicted and target boxes.
                - DetectionLoss - Combines classification + box loss with optional L2 regularization.
                - IoUWeightedBoxLoss - Weights box loss based on IoU confidence.

                Purpose :
                - Improves training stability and accuracy.
                - Supports multi-scale detection with per-level loss aggregation.
                - Clips gradients and normalize predictions.

                Handles :
                - Binary cross-entropy + modulation (Focal).
                - IoU computation + safe division.
                - Box regression smoothing and weighting.
                - Regularization and loss clipping.

                Functions :
                - _init_ -> Set configs (alpha, gamma, etc.).
                - forward -> Computes loss per class, box, or both.

                Outputs :
                - { 'loss', 'cls_loss', 'box_loss', 'reg_loss' } per batch.



        / tests

            . test_anchor_generator.py - Anchor Generator Unit Test & Visualizer.

                - Tests and visualizes the output of the AnchorGenerator class to ensure anchor boxes are being correctly generated and distributed across features maps.

                Purpose :
                - Validates the shapes, dimensions, and distribution of generated anchors.
                - Visually inspects anchor layouts and size distributions for debugging.
                - Provides warnings for invalid or degenerate anchor boxes.

                Handles :
                - Creates dummy FPN-style feature maps.
                - Calls AnchorGenerator to generate anchors for all levels.
                - Plots anchors over feature map grid.
                - Plots anchor area vs. aspect ratio scatter for inspection.

                Functions :

                - test_anchor_generator -> Main test logic and visualizer loop.
                
                - analyze_anchor_box -> Checks each anchor for validity and stats.

                - visualize_anchors_detailed -> Plots grid-aligned anchors over image bounds, shows size distribution, generates per-level analysis.


            . test_augmentation.py - Augmentation Unit Test.

                - Tests our TorchVision-based augmentation pipeline in isolation by applying transforms to a synthetic image & visual outputs.

                Purpose :
                - Validates DetectionAugmentation class correctly transforms image & bounding box inputs.
                - Visualizes how bounding boxes change under augmentation.
                - Saves Before/After Images for Inspection.

                Handles :
                - Creating synthetic test image and boxes.
                - Draws and saves bounding boxes pre- and post-augmentation.
                - Applies both train and val transforms from DetectionAugmentation.
                - Denormalizes and visualizes image results.

                Functions :
                
                - test_augmentation -> Generates test image w/ labeled bounding boxes, applies train and val transforms, saves visualized output. 

                Output :
                - Augmented Images w/ Bounding Boxes saved in test_results/augmentation.
                - Debug Console Output with Shape, Pixel Range, and Box Summaries.

            . test_dataset.py - Dataset Unit Test + Visualization

                - Tests SKU-110K Dataset Loading & Visualization Pipeline. Applies basic augmentations, checks for bounding box validity, and saves output images for inspection.

                Purpose :
                - Ensures dataset is loading and preprocessing images and annotations correctly.
                - Verifies bounding box ranges and formats before and after augmentation.
                - Visualize real data samples (original and augmented) w/ annotations.
                
                Handles :
                - Initializes SKU110KDataset & DetectionAugmentation.
                - Samples Random Dataset Entries.
                - Applies visualizations and saves output images.
                
                Functions :

                - load_config -> Loads YAML file for test configuration.

                - visualize_sample -> Draws bounding boxes on images, saves or displays them.

                - analyze_boxes -> Compute width, height, and area stats for bounding boxes.

                - test_dataset_loading -> Loads samples, Visualizes original images, Applies augmentations, Visualizes tranformed results.

                Output :
                - Visualizations saved to test_results/dataset/
                - Prints console info for Image Shape, Box Count and Value Ranges, and Augmentation Results.
                
            . test_detection_head.py - Unit Test for Detection Head.

                - Tests foward pass and anchor matching behavior of the detection head on synthetic data.

                Simple Breakdown :

                - 1. Creates fake feature maps and targets.
                - 2. Runs the detection head's forward pass.
                - 3. Matches anchors to targets.
                - 4. Computes classification and box regression losses.
                - 5. Print debug info and loss values.

                Purpose :
                - Validates that the detection head can run end-to-end on mock inputs.
                - Ensure anchor matching and loss functions work as expected.
                - Helps catch shape errors or invalid outputs early.

                Handles :
                - Synthetic input generation.
                - Invokes model forward and loss functions.
                - Prints shapes, matches, and loss values for debugging.

                Functions :
                - test_detection_head -> Simulates a mini-batch, runs prediction + matching + loss, and print stats.

                Output :
                - Printed logs with Feature Map Dimensions, Anchor Stats, Loss Values, Match Counts and Warnings.

            . test_detector.py - Sanity Check for Complete Detection Pipeline.

                - Runs a full test on teh end-to-end ObjectDetector using dummy inputs.

                Simple Breakdown :
                - 1. Load model config from config.yaml.
                - 2. Build full detector with Backbone, FPN, and Detection Head.
                - 3. Run inference on random images at different resolutions.
                - 4. Checks output shapes of feature maps and predictions.

                Purpose :
                - Validates structural integrity of the full detection model.
                - Ensure all components work together.
                - Verifies input/output sizes at each stage.

                Functions :
                - verify_feature_maps -> Confirms feature map resolution is reasonable.
                - test_detector -> Builds the detector, runs it on dummy inputs, and prints output shapes for backbone, FPN, and detection head.

                Outputs :
                - Prints shapes of backbone, FPN, and head outputs.
                - Final message : "All tests passed!" if everything works.

            . test_fpn.py - Unit Test for Feature Pyramid Network.
            
                - Validates that FPN outputs have correct shapes, channels, and spatial alignment.

                Simple Breakdown :
                - 1. Generate fake backbone features for each layer.
                - 2. Pass them through the FPN.
                - 3. Checks that outputs match expected shapes and channels.
                - 4. Confirm spatial sizes match original inputs.
                - 5. Print FPN output stats and confirm successes.

                Purpose :
                - Ensures the FPN correctly processes multiscale features from the backbone.

                Handles :
                - Creates dummy inputs tensors mimicking layer1 to layer4 of ResNet.
                - Verifies output FPN levels have corerct channel count and matchign spatial size with respective backbone layers.

                Functions :
                - test_fpn -> Tests FPN output structure and dimensions using dummy feature maps.

            . test_losses.py - Unit Tests for Detection Loss Functions.
                
                - Validates correctnessa nd behavior of all loss modules in losses.py

                Simple Breakdown :
                - Tests Focal Loss w/ multi-class prediction and checks gradient flow.
                - Tests IoU Loss on valid (x1, y1, x2, y2) boxes and range of values.
                - Tests combined DetectionLoss with dummy multi-scale outputs and targets.
                - Tests IoU-weighted Smooth L1 loss using dummy IoU values.

                Purpose :
                - Ensure each loss returns valid, non-zero, differentiable tensors.
                - Validate loss keys and shapes in detection pipeline.
                - Prevent silent errors during training via shape checks and value ranges.

                Structure :
                - TestLosses(unittest.TestCase) -> Four test methods, one per loss type.


            . test_training.py - Unit & Integration Tests for Training Workflow.

                - Tests optimizer, scheduler, trainer, and end-to-end training with dummy and real datasets.
                
                Simple Breakdown :
                - 1. Validates optimizer grouping and weight decay logic.
                - 2. Checks learning rate progression during warmup and cosine decay.
                - 3. Uses dummy dataset / model to simulate training end-to-end.
                - 4. Confirms detection loss integration and checkpoint saving.
                - 5. Runs actual SKU-110K dataset through a shortened training cycle.
                - 6. Validates detection metrics like IoU and positive anchor ratios.

                Dependencies :
                - src / training / optimizer.py
                - src / training / scheduler.py
                - src / training / trainer.py
                - src / utils / losses.py
                - src / utils / metrics.py
                - src / data / dataset.py
                - src / utils / augmentation.py
                - src / model / detector.py

                Purpose :
                - Ensure core training componetns work independently and together.
                - Detects regressions in LR handling, loss calculation, checkpoint saving, and data processing.

                Functions :
                - test_optimizer -> Test optimizer building logic and correct aprameter group handling.
                - test_scheduelr -> Test learning rate warmup and cosine decay using WarmupCosineScheduler.
                - test_trainer -> Test full training flow using a dummy dataset and dummy model. Validates checkpoint saving.
                - test_detection_metrics -> Verifies detection metric calculations like mean IoU and positive ratio.
                - test_training_pipeline -> Test entire training pipeline with a small subset of SKU110K dataset and real model.

                Classes :
                - DummyDataset -> Custom Dataset that returns random tesnors simulating image boxes and lables.
                - DummyModel -> Simulates CNN-based object detection model outputting dummy cls and bbox predictions for test purposes.

            


    / training 

        . optimizer.py - Builds & Configures the Optimizer.

            - Creates an optimizer to update model weights during training.

            Simple Breakdown :
            - 1. Separate model parameters into decay vs. no-decay groups.
            - 2. Choose optimizer type (adam, adamw, sgd).
            - 3. Return a configured optimizer with correct hyperparameters.

            Purpose :
            - Applies weight decay to the right parameters (ignores biases and norms).
            - Initializes and returns the optimizer with training configurations.

            Handles :
            - Parameter grouping for proper weight decay handling.
            - Supports Adam, AdamW, and SGD optimizers.
            - Validates parameter separation to avoid training bugs.

            Functions :
            - OptimizerBuilder.build -> Returns an optimizer with grouped parameters for decay / no-decay.
            - build_optimizer -> Quick builder for Adam, AdamW, or SGD based on Config Directory.

        . scheduler.py - Learning Rate Scheduling during Training.
            
            - Adjusts the learning rate over time using warmup and decay strategies to improve convergence.
            
            Simple Breakdown :
            - 1. Start with a low learning rate (warmup).
            - 2. Gradually increase to a peak.
            - 3. Slowly decrease using cosine or step decay.

            Purpose :
            - Prevents unstable training early on.
            - Improves final model performance through smart LR adjustment.
            - Supports OneCycle, Cosine, Step, and Plateau Schedulers.

            Handles :
            - Custom Warmup + Cosine Scheduelr (WarmupCosineScheduler).
            - PyTorch built-in schedulers liek OneCycleLR, StepLR, and ReduceLRonPlateau.
            - Scheduler type is selected via config.

            Functions / Classes :
            - WarmupCosineScheduler -> Increases LR linearly, then decays it with cosine.
            - get_lr() -> Computes LR based on current epoch.
            - build_scheduler() -> Builds and returns a scheduler based on config type (e.g., cosine, onecycle, step, plateau).

        . trainer.py - Training Engine for Detection Model.
            
            - Manages the full training pipeline : loading data, optimizing model, computing metrics, and saving checkpoints.

            Simple Breakdown :
            - 1. Load datasets and prepare data loaders.
            - 2. Train model per epoch with loss + metric tracking.
            - 3. Validate after each epoch to check performance.
            - 4. Save best and periodic checkpoints.
            - 5. Supposrrts custom collate, mAP/F1 metrics, debug handling.

            Dependencies :
            - 

            Purpose :
            - Streamlines training logic for object detection models.
            - Tracks loss, mAP, F1, and handles gradient clipping.
            - Saves models conditionally.
            - Integreates optimizer, scheduler, visualizer, and metrics.

            Handles :
            - Train and validation loading via DataLoader.
            - Optimizer and LR scheduler Building.
            - Epoch-wise training,
            - Evaluation.
            - Loss calculation from predicted and ground truth boxes/labels.
            - Logging metrics like mAP and F1 from IoU comparisons.

            Functions / Classes:
            - Trainer ->  Central class managing training and validation.
            - train_epoch() -> Trains one full epoch, logs and tracks losses + metrics.
            - validate() -> Runs model on validation set and computes metrics.
            - train() -> Full training loop over all epochs.
            - save_checkpoint() ->  Saves model weights based on performance.
            - compute_loss() -> Computes total loss using classification and box regression.
            - compute_map() ->  Computes mean average precision using 11-point interpolation.
            - compute_f1() ->  Computes F1 score using predicted vs. ground truth IoUs.
            - collate_fn() ->  Custom batch collate function for object detection data. 

    / utils 

        . augmentation.py - Augmentation Pipeline for Object Detection.

            - Defines image augmentation logic for training and validation phases, ensuring bounding boxes remain accurate and normalized.

            Simple Breakdown :
            - 1. Resize and normalize input images for both training and validation.
            - 2. Apply additional augmentations during training.
            - 3. Convert bounding boxes from normalized format to pixel coordinates before augmentation.
            - 4. Convert augmented bounding boxes to normalized format after tnraformation.
            - 5. Perform rangoe and validity checks on bounding box coordinates.

            Purpose :
            - Preprocess images and bounding boxes for object detection models.
            - Ensure bounding boxes stay aligned and within valid ranges after data augmentation.
            - Enable stronger generalziation during training through varied image transofmrations.

            Handles :
            - Converts bounding boxes between normalized and pixel formats before / after transformation.
            - Applies Albumentations transforms consistently for both training and validation pipelines.
            - Includes debug loggin to verify box transformation integrity.
            - Perfomrs strict validation of fina lbounding box coordinates.

            Functions :
            - init -> Sets up transformation pipelines and target image dimensions.
            - _call_ -> Applies the appropriate transform, adjusts boundign boxes, and ensures validity post-augmentation.

            Outputs :
            - Dictionary :
                - image -> Transformed image tensor.
                - bboxes -> Transformed and re-normalized bounding boxes.
                - labels -> Class labels associated with each bounding box.

        . box_ops.py - Bounding Box IoU Comparison.

            - Defines a utility function to compute Intersection over Union between two sets of bounding boxes.

            Simple Breakdown :
            - 1. Accepts two sets of boxes : predictions and ground truth.
            - 2. Computes individual box areas.
            - 3. Calculates pairwise intersection areas between all box pairs.
            - 4. Computes union areas from individual areas and intersections.
            - 5. Retursn an [N x M] IoU Matrix. 

            Purpose :
            - Measure the overlap between predicted and ground truth bounding boxes. 
            - Provide IoU values used for : Anchor Matching during Training, Eval Metrics like mAP and F1 score.

            Handles :
            - Supports variable sized inputs : N predictions and M ground truth boxes.
            - Efficient vectorized computation using PyTorch tensors.

            Functions :
            - box_iou (boxes1, boxes2) -> Returns IoU matrix [N, M] for each box pair.

            Outputs :
            - torch.Tensor : A [N x M] matrix where each element represents the IoU between a box in boxes1 and a box in boxes2.

        . metrics.py - Anchor Matching Metrics.

            - Provides uitlity functiosn to evaluate the quality of anchor-to-ground-truth matches during training using IoU-based metrics.

            Handles :
            - Computes matching quality and statistics based on positive matches.
            - Supports both NumPy arrays and PyTorch tensors.
            - Handles cases with no positive matches safely.

            Functions :
            - compute_matching_quality -> Returns metrics like mean IoU, postivie ratio, and IoU range for positive matches.
            - compute_matching_statistics -> Provides extended stats including total anchor, std IoU, and min/max values.

            Output :
            - Returns a dictionary of float metrics for analyzing detections.

        . visualization.py - Visualization Tools for Object Detection.

            - Provides a suite of visualization tools for object detection tasks, including image predictions, ground truth comparisons, anchor visualizations, and IoU-based diagnostics.

            Handles :
            - Display of predicted vs. ground truth bounding boxes.
            - Visualization of feature maps from backbone or FPN layers.
            - Anchor and matched-anchor overlays with IoU metrics.
            - Histograms and bar charts for detection metrics.

            Functions :
            - visualize_batch : Displays side-by-side images showing predicted and ground truth boxes with confidence scors.
            - visualize_feature_maps : Visualizes selected channels of feature maps for each layer.
            - visualize_anchors : Shows anchor boxes from different FPN levels overlaid on the image.
            - visualize_matched_anchors : Enhanced diagnostic showing matched anchors, IoU scores, and detection quality metrics in grid layout.

            Output :
            - Matplotlib visualizations for model debugging and performance assessment.

    train.py 

    evalute.py 