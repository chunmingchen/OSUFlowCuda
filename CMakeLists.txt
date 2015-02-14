# use mpi: configure with cmake CXX=mpicxx
#          with GUI: when creating project specify compiler
cmake_minimum_required (VERSION 2.8)
project (osuflowCuda)

FIND_PACKAGE(CUDA)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(osuflow)

# settings

# settings for large data
add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE=1)

# directories
INCLUDE_DIRECTORIES( 
	${INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/../common
	# cuda
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    #${CUDA_SDK_ROOT_DIR}/common/inc  # no need
	)
LINK_DIRECTORIES( 
	${LINK_DIRECTORIES}
	${CMAKE_SOURCE_DIR}/../osuflow/src
	${lib_path}
	# cuda
	${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32
	#${CUDA_SDK_ROOT_DIR}/common/lib/Win32
	)

# targets
CUDA_ADD_LIBRARY(${PROJECT_NAME}
	OSUFlowCuda.C
	streamline_kernel.cu
	)


# test
add_executable(testReadFlowfield testReadFlowfield.cpp)
target_link_libraries(testReadFlowfield OSUFlow ${CUDA_LIB} ${PROJECT_NAME} )

# example
add_executable(StreamlineSimpleCuda StreamlineSimple.C)
target_link_libraries(StreamlineSimpleCuda   ${PROJECT_NAME} OSUFlow )

# example with OpenGL
find_package(OpenGL)
find_package(GLUT)

include_directories(
	${GLUT_INCLUDE_PATH}
	${OPENGL_INCLUDE_DIR}
	${GUL_INCLUDE_PATH}
)

add_executable(glStreamlineSimpleCuda glStreamlineSimple.C)
target_link_libraries(glStreamlineSimpleCuda OSUFlow ${CUDA_LIB} ${PROJECT_NAME} ${CUDA_LIB} ${OPENGL_LIBRARY} ${GLU_LIBRARY} ${GLUT_LIBRARY} )

