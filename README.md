# CTSAE

> Code of paper 'Cross-Temporal Spectrogram Autoencoder (CTSAE): Unsupervised Dimensionality Reduction for Clustering Gravitational Wave Glitches' (CVPRW 2024)

### [arXiv](https://arxiv.org/abs/2404.15552) 



>**Abstract:** The advancement of The Laser Interferometer Gravitational-Wave Observatory (LIGO) has significantly enhanced the feasibility and reliability of gravitational wave detection. However, LIGO's high sensitivity makes it susceptible to transient noises known as glitches, which necessitate effective differentiation from real gravitational wave signals. Traditional approaches predominantly employ fully supervised or semi-supervised algorithms for the task of glitch classification and clustering. In the future task of identifying and classifying glitches across main and auxiliary channels, it is impractical to build a dataset with manually labeled ground-truth. In addition, the patterns of glitches can vary with time, generating new glitches without manual labels. In response to this challenge, we introduce the Cross-Temporal Spectrogram Autoencoder (CTSAE), a pioneering unsupervised method for the dimensionality reduction and clustering of gravitational wave glitches. CTSAE integrates a novel four-branch autoencoder with a hybrid of Convolutional Neural Networks (CNN) and Vision Transformers (ViT). To further extract features across multi-branches, we introduce a novel multi-branch fusion method using the CLS (Class) token. Our model, trained and evaluated on the GravitySpy O3 dataset on the main channel, demonstrates superior performance in clustering tasks when compared to state-of-the-art semi-supervised learning methods. To the best of our knowledge, CTSAE represents the first unsupervised approach tailored specifically for clustering LIGO data, marking a significant step forward in the field of gravitational wave research.

If you find this work useful, please cite: 

```
@misc{li2024crosstemporalspectrogramautoencoderctsae,
      title={Cross-Temporal Spectrogram Autoencoder (CTSAE): Unsupervised Dimensionality Reduction for Clustering Gravitational Wave Glitches}, 
      author={Yi Li and Yunan Wu and Aggelos K. Katsaggelos},
      journal={CVPRW},
      year={2024}
      url={https://arxiv.org/abs/2404.15552}, 
}
```


## Installation
```
conda env create -f environment.yml
```



## Training CTSAE on LIGO O3 dataset
To reproduce the result present in our paper and train our model on LIGO O3 dataset

1. Run the GravitySpy pipeline https://github.com/Gravity-Spy/gravityspy-ligo-pipeline to obtain gravitational wave spectrograms, and organize these spectrograms into the following structure:
```
root_directory
├──sub_0.5
├──sub_1.0
├──sub_2.0
└──sub_4.0
        ├── 1080Lines
                ├── H1_0Co6t54xXL_spectrogram_4.0.png
                ├── H1_0EwreMqouX_spectrogram_4.0.png
                └── ...
        ├── 1400Ripples
        ├── Air_Compressor
        └── ...  

```

2.  Run ```process_data.py``` to process the spectrograms 

3. Run ```train.sh``` to train the model


  


## Testing CTSAE
 Run ```extract.py``` to extract the latent code and reduce the spectrogram into a low-dimensional space

## Acknowledgement

Part of the code is based on [Conformer](https://github.com/pengzhiliang/Conformer).



