# Opening the Black Box

## Heatmap Clustering to Understand the Misbehaviours Exposed by Automatically Generated Test Inputs

Deep Neural Networks (DNNs) revolutionize the automated image classification field, demonstrating impressive performances on large benchmark datasets. DNNs can also be applied to other fields relevant to the research and the industry, such as natural language processing, human action recognition, and physics.

Despite the high performances of DNNs, they essentially act as black boxes. In fact, they have an intrinsic drawback as it is unclear how and why they arrived at a particular decision.  The lack of transparency of these models is a severe disadvantage, as it prevents human experts from verifying, interpreting, and understanding their reasoning.

DNNs are essentially acting as black boxes.

Explainable AI (XAI) techniques aim to explain DNN decisions so that they are interpretable by humans.

Heatmaps are a common approach adopted by XAI techniques in the image classification field. The idea is to capture and represent the importance of pixels in image predictions in an interpretable form. Layer-wise Relevance Propagation (LRP) is a widely used method to generate such heatmaps. LRP works by computing scores for image pixels, denoting the impact on the final decision.

Feature maps are a novel approach to explain DNNs proposed in the Software Engineering literature. The feature map provides a human-interpretable picture of how different features affect the system behaviour and performance. This strategy requires identifying and quantifying the dimensions of the feature space for a given domain.

The thesis aims to tackle the current transparency issue of DNN systems, investigate the similarities and differences between heatmaps and feature maps, make them more interpretable, and consider possible integrations of the two.

The work is composed of two steps. The first one involves applying clustering techniques to XAI heatmaps and feature maps, making them more interpretable and comparable. The second one is empirically comparing the obtained clusters. 