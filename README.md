# 3D implicit surface model from sparse point-clouds generated from Synthetic aperture radar measurements
We utilize the Signed distance function (SDF) to represent the surface of objects. The SDF network is based on
1. DeepSDF - 
 Park, Jeong Joon, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. "Deepsdf: Learning continuous signed distance functions for shape representation." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 165-174. 2019.
2. Multi-view surface reconstruction
Yariv, Lior, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and Yaron Lipman. "Multiview neural surface reconstruction by disentangling geometry and appearance." Advances in Neural Information Processing Systems 33 (2020): 2492-2502.

The surface is iteratively refined and resampled to produce iso-points using resampling methods on the surface representation to get robust surface representation. The iso-points method extracts uniformly sampled points on the surface to get smooth surface representation using the method described in 
Yifan, Wang, Shihao Wu, Cengiz Oztireli, and Olga Sorkine-Hornung. "Iso-points: Optimizing neural implicit surfaces with hybrid representations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 374-383. 2021.
