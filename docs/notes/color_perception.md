# Color Perception

We explore the intricacies of light, color perception, and the advanced technologies used to replicate the vibrant colors we see in the natural world. This note will highlight the foundational principles of color and its interactions with the human visual system, and how these ideas are applied to contemporary digital displays.

## What is Color

Color is a perceptual phenomenon arising from the human visual system's interaction with light. Color is essentially a "side effect" created by our brain when specific wavelengths of light are emitted, reflected, or transmitted by objects. However, humans can only percieve color within a certain range of the electromagnetic spectrum, from around 300 to 700 nanometers, and each wavelength can correspond to a different color  ([see more about light here](../course/computational_light.md)). 

Because waves at different frequencies do not constructively or destructively interfere, it is interesting to see how our brain can process multiple frequencies of light simultaneously.

## Biological Foundations of Perceiving Color
The perception of color originates from the absorption of light by photoreceptors in the eye, converting the light into electrical signals to be interpreted by the brain. Much of the brain's color interpretation is due to the wavelength, intensity, or context viewed, causing rich and different experiences of color.

To further elaborate, these photoreceptor cells in the eye are called rods and cones. Rods, which are relatively more common in the periphery, help people see in low-light conditions, but can only interpret in a greyscale manner. Cones, which are more dense in the fovea, are pivotal in color perception in relatively normal environments. The cones are categorized into three types based on their sensitivity to specific wavelengths of light, corresponding to long (L), medium (M), and short (S) wavelengths. These three types of cones allow us to better understand the trichromatic theory, which posits that human color perception stems from combining stimulations of the LMS cones, which correspond to red, green, and blue light, respectively. For example, red is perceived when L cones are significantly stimulated more than the other types, and blue is perceived when S cone activation is more prominent.

| <img src="https://www.ncbi.nlm.nih.gov/pmc/articles/instance/4167798/bin/nihms595119f1.jpg" width="640" alt/> |
|:--:| 
| *Spectral Sensitivities of LMS cones*|

However, the story of color perception deepens with the concept of color opponency. This theory reveals that our perception of color is not just a matter of additive combinations of primary colors but also involves a dynamic interplay of opposing colors: red versus green, blue versus yellow. This phenomenon is rooted in the neural pathways of the eye and brain, where certain cells are excited or inhibited by specific wavelengths, enhancing our ability to distinguish between subtle shades and contrasts.

The interplay between rods and cones is a healthy balance between light sensitivity and color detection. The transition between these modes is seamless but so complex, demonstrating the versatility of our visual system.

Here are useful resources behind this topic:
* `B. P. Schmidt, M. Neitz, and J. Neitz, "Neurobiological hypothesis of color appearance and hue perception," J. Opt. Soc. Am. A 31(4), A195–207 (2014)`

## Display Realism (What does it mean to be realistic)

When considering the realism of displays, it is important to define what realism entails in the context of color perception. If we were to have a display, disregarding all cost and engineering challenges, just solely built to be "lifelike", what would we need to achieve? To answer this question, we would need to be able to apply the complex principles of human color perception and display technologies. 

1. Accurate Reproduction of Colors (or at least perceptible)
The most important characteristic of a realistic display is to accurately reproduce color. Current display technologies combine three color primaries (Red, Green, Blue) in different intensities attempting to recreate large ranges of color called a color space. It is possible to choose different primary colors, or even the number of primaries to represent one's color space, but its efficacy can be expressed by how vast the resulting color space is. The human color gamut is a collection of all visible human lights, and is currently impossible to represent with only three primaries. Because the gamut is continuous and infinite, you would need an infinite amount of primaries to represent all colors.

Fortunately, one promising solution is the use of **metamers**— applying different combinations of wavelengths that produce the same color perception in the human eye. This means two separate colors can ellicit the same LMS cone response as each other. This allows displays to recreate a vast range of colors on a limited set of primaries. 
!! TODO: talk about generating the color space with primaries 

2. Accounting for Photopic vs Scotopic vision
Human perception is extremely context dependent, where we need to adapt to various lighting conditions like low-light (scotopic) and lit (photopic) scenes. Displays must be able to figure out how to preserve the rod and cone functionality under all these different environments.

3. Chromaticity + Brightness


## Conclusion
As we dive deeper into light and color perception, it becomes evident that the task of replicating the natural spectrum of colors in technology is still an evolving journey. This exploration into the nature of color sets the stage for a deeper examination of how our biological systems perceive color and how technology strives to emulate that perception.