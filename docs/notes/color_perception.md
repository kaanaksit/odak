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

## Intersection of Color Perception and Display Technologies

There are several different display technologies trying to accurately replicate the spectrum of colors that our eyes percieve. Such displays include Light-Emitting Diodes (LEDs), Liquid Crystal Displays (LCDs), Organic LEDs (OLEDs), and much more. Their spectra can be analyzed using the principles of the RGB color model to stimulate our cones similar to natural light. This can be graphically represented with spectral power distribution (SPD) graphs, showing relationships between what colors we can perceive (wavelengths) versus the power of the light source. This could be used to analyze color characteristics of different displays.

### Light Emitting Diode Displays: 
|  |
|:--:| 
| *Placeholder for SPD of LED*|

LED (Light Emitting Diode) displays use individual LEDs as pixels or as a backlight for LCD panels. LED displays produce colors by adjusting the intensity of red, green, and blue LEDs, closely aligning with the way human eyes perceive color through trichromatic vision.

### Liquid Crystal Displays (LCDs)
|  |
|:--:| 
| *Placeholder for SPD of LCD*|

LCD has a backlight to illuminate pixels, with liquid crystals controlling the passage of light through RGB filters.


### Organic Light-Emitting Diodes (OLEDs)
|  |
|:--:| 
| *Placeholder for SPD of OLEDs*|
OLED displays have each pixel produce their own light, allowing for a broader color spectrum.



## Conclusion
As we dive deeper into light and color perception, it becomes evident that the task of replicating the natural spectrum of colors in technology is still an evolving journey. This exploration into the nature of color sets the stage for a deeper examination of how our biological systems perceive color and how technology strives to emulate that perception.