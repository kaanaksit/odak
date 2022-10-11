# What is Odak?
Odak (pronounced "O-dac") is the fundamental library for scientific computing in optical sciences, computer graphics and visual perception.

## Why does it exist?
This question has two answers. 
One of them is related to the history of `Odak`, which is partially answered in the next section.
The other answer lies in what kind of submodules `Odak` has in it.
Depending on a need of a scientist at all levels or a professional from the industry, these submodules can help the design processes in optics and visual perception.

Odak includes modules for geometric [3D raytracing](https://github.com/kunguz/odak/tree/master/odak/raytracing/), [Jones calculus](https://github.com/kunguz/odak/tree/master/odak/jones), [wave optics](odak/wave), and [a set of tools](https://github.com/kunguz/odak/tree/master/odak/tools) to ease pain in [measurement](https://github.com/kunguz/odak/tree/master/odak/measurement), [exporting/importing CAD](https://github.com/kunguz/odak/tree/master/odak/tools/asset.py), and [visualization](https://github.com/kunguz/odak/tree/master/odak/visualize) during a design process.
We have generated a set of recipes that go well with machine learning approaches compatible with the PyTorch learning framework as provided [here](https://github.com/kunguz/odak/tree/master/odak/learn).
We have created many [test scripts](https://github.com/kunguz/odak/tree/master/test/) to inspire how you use Odak and helping your design process.
Finally, we have created a [distribution system](https://github.com/kunguz/odak/tree/master/odak/manager) to process tasks in parallel across multiple computing resources within the same network.
Odak can either run using CPUs or automatically switch to [NVIDIA GPUs](https://github.com/kunguz/odak/tree/master/odak/__init__.py#L8).

## History
In the summer of 2011, I, [Kaan Akşit](https://kaanaksit.com), was a PhD student.
At the time, I had some understanding of the Python programming language, and I created my first [Python based computer game](https://www.youtube.com/watch?v=r9RIzKCGrmU) using `pygame`, a fantastic library, over a weekend in 2009.
I was actively using Python to deploy packages for the Linux distribution that I supported at the time, [Pardus](https://distrowatch.com/table.php?distribution=pardus).
Meantime, that summer, I didn't have any internship or any vital task that I had to complete.
I was super curious about the internals of the optical design software that I used at the time, `ZEMAX`.
All of this lead to an exciting never-ending excursion that I still enjoy to this day, which I named Odak.
`Odak` means focus in Turkish, and pronounced as `O-dac`.

The very first paper I read to build the pieces of Odak was `General Ray tracing procedure" from G.H. Spencer and M.V.R.K Murty`, an article on routines for raytracing, published at the [Journal of the Optical Society of America, Issue 6, Volume 52, Page 672](https://doi.org/10.1364/JOSA.52.000672).
It helped to add reflection and refraction functions required in a raytracing routine.
I continuously add to Odak over my entire professional life.
That little raytracing program I wrote in 2011 is now a vital library for my research, and much more than a raytracer.

I can write pages and pages about what happened next.
You can accurately estimate what happened next by checking [my website and my cv](https://kaanaksit.com).
But I think the most critical part is always the beginning as it can inspire many other people to follow their thoughts and build their own thing!
I used Odak in my all published papers.
When I look back, I can only say that I am thankful to 2011 me spending a part of his summer in front of a computer to code a raytracer for optical design.
Odak is now more than a raytracer, expanding on many other aspects of light, including vision science, polarization optics, computer-generated holography or machine learning routines for light sciences.
Odak keeps on growing [thanks to a body of people that contributed over time](https://github.com/kunguz/odak/blob/master/THANKS.txt).
I will keep it growing in the future and will continually transform into the tool that I need to innovate.
All of it is free as in free-free, and all is sharable as I believe in people.
