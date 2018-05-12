# Laifu to Waifu
This is a project that was originally created at [Science Hack Day SF 2017](http://sf.sciencehackday.org/) (where it won the People's Choice award), and is still being updated and improved after the competition ended. The objective is to take images of human faces and turn them into anime images.

## Method
This model uses [CycleGAN](https://github.com/junyanz/CycleGAN) to translate images from one style to the other.

The anime face dataset was gathered using the data collection method proposed in [https://arxiv.org/abs/1708.05509](https://arxiv.org/abs/1708.05509)

To closely match the makeup of the anime dataset, the human faces were sourced from the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), filtered by [female, attractive]

The model is based on the Tensorflow implementation of CycleGAN found here:
https://github.com/xhujoy/CycleGAN-tensorflow

## Results
The model is still being improved but this is the best example so far:

![Waifu Generator](https://i.imgur.com/0v8R5Vl.png)

Left Image: Input<br />
Right Image: Output

## Why
For Science
