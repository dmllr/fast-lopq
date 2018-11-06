add_executable(fast-lopq-test
	main.cpp
	argsparser.cpp
	argsparser.h
	device.cuh
	device.cu
)

target_link_libraries(fast-lopq-test
	fast-lopq
)

set_property(TARGET fast-lopq-test
	PROPERTY CUDA_SEPARABLE_COMPILATION ON
)
