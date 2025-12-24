# Bridging Language and 3D Assets
### Comparing Embedding-Based and Contrastive Approaches for Text-to-3D Retrieval

Adam Rolander	| Pablo Calderon | Juan Fernández de Navarrete | Gayathri Kuniyil	| Arjun Kohli | Ethan Cota

### Abstract
3D asset libraries have grown tremendously, making effective text-to-3D retrieval a valuable tool in 3D content creation. However, most asset retrieval systems struggle with the semantic variability of natural language queries. We address the question of whether an unsupervised embedding-based approach or a supervised contrastive approach yields better semantic alignment between text descriptions and corresponding 3D assets. We implement a baseline unsupervised retrieval pipeline using a pre-trained sentence encoder (MiniLM L6 V2) and compare its performance against a supervised multimodal model fine-tuned on text-image pairs using a contrastive learning objective. To evaluate the role of linguistic diversity, the supervised model was trained on both base (1 caption : 1 asset) and paraphrase-augmented datasets (11 captions : 1 asset, synthesized using Gemini-2.5-Flash). All models achieved perfect Recall@10 on original captions. However, evaluation under linguistic perturbations revealed key differences: the Supervised Base model suffered from reduced stability (R@10=0.923) and poor discrimination (Discrimination=1.00—maximum false positives), indicating severe overfitting. The Supervised Augmented model recovered perfect stability (R@10=1.00) and significantly improved discrimination (Discrimination=0.50), demonstrating the necessity of augmentation. Notably, the Unsupervised Baseline showed the highest discrimination power (Discrimination=0.25). Our findings conclude that unsupervised embedding-based retrieval is a robust baseline in low-data regimes, but supervised multimodal models, when trained with paraphrastic augmentation, achieve superior semantic grounding and stability necessary for practical text-to-3D search systems.

**Code:** https://github.com/AdamRolander/3D-Asset-Retrieval 

**Model Weights:** https://huggingface.co/Aerolandaz/text_to_3D_retrieval/tree/main 
