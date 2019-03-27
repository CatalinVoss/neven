This is the face detection library extracted from Android source tree. You can now use it in non-Android applications.

It was originally written by Neven Vision, a company acquired by Google in 2006.

Building
--------

$ make
$ sudo make install


Writing Application
-------------------

It's easy to use this library in your applications.

```
	#include "neven.h"
	...
	struct neven_env *env = neven_create(width, height, max_faces);
	int num_faces = neven_detect(env, buffer); // 'buffer' points to a grayscale 1-byte-per-pixel raw image buffer
	int i;
	for (i = 0; i < num_faces; i++)
	{
		struct neven_face face;
		neven_get_face(env, &face, i);
		printf("face %d is at (%f, %f)\n", i, face.midpointx, face.midpointy);
	}
	neven_destroy(env);
```

OpenCV 2+ examples
------------------

`opencv/neven_detector.hpp` and `opencv/neven_detector.cpp` contain an examples for building an object oriented wrapper.
