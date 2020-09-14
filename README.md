# semantic_image_segmentation
Binary Image Segmentation on Pascal VOC 2012


Dataset comprises of down sampled images from Pascal VOC 2012.

Three models are used to perform binary (background - forground (all classes combined) on the data.
1) Baseline: FCN model comprised of te first five layers of Alexnet with minor modifications, trained using BCE loss.
2) SVM model trained using hinge loss from features extracted from baseline model
3) Structured SVM model employing Pott's model to capture relationship between neighbouring pixels. Trained using hamming-loss augmented structured hinge loss. 
