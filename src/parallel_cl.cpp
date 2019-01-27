#include <float.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sstream>
#include <iostream>
#include <fstream>

#include "src/dijkstra.hpp"


///
//  Globals
//
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

///
//  Macros
//
#define check_error(a, b) assert_msg(a, b, __FILE__ , __LINE__)

///
//  Macro Options
//
#define NUM_ASYNCHRONOUS_ITERATIONS 10  // Number of async loop iterations before attempting to read results back

///
//  Function prototypes
//
bool maskArrayEmpty(int *maskArray, int count);

cl_device_id get_first_device(cl_context cxGPUContext);

static inline void assert_msg(int errNum, int expected, const char* file, const int lineNumber);
int roundWorkSizeUp(int groupSize, int globalSize);

void print_device_info()
{
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;
    cl_uint maxWorkItemDimensions;
    size_t * maxWorkItemSize;
    size_t maxWorkGroupSize;

    printf("OpenCL Device info:\n");
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (auto i = 0U; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (auto j = 0U; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s (id: %p)\n", j+1, value, devices[j]);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
            printf(" %d.%d Max work gorup size: %zu\n", j+1, 5, maxWorkGroupSize);

            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                    sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, NULL);
            printf(" %d.%d Max work item dimension: %u\n", j+1, 6, maxWorkItemDimensions);

            maxWorkItemSize = new size_t[maxWorkItemDimensions];
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    sizeof(size_t) * maxWorkItemDimensions, maxWorkItemSize, NULL);
            for (auto k = 0U; k < maxWorkItemDimensions; ++k)
            {
                printf(" %d.%d.%d Max work item size: %zu\n", j+1, 7, k + 1, maxWorkItemSize[k]);
            }


            delete[] maxWorkItemSize;
        }

        free(devices);

    }

    free(platforms);
}

static cl_program load_and_build_program(cl_context opencl_context, const char *fileName)
{
    pthread_mutex_lock(&mtx);

    cl_int errNum;
    cl_program program;

    // Load the OpenCL source code from the .cl file
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *source = srcStdStr.c_str();

    check_error(source != NULL, true);

    // Create the program for all GPUs in the context
    program = clCreateProgramWithSource(opencl_context, 1, (const char **)&source, NULL, &errNum);
    check_error(errNum, CL_SUCCESS);
    // build the program for all devices on the context
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        char cBuildLog[10240];
        clGetProgramBuildInfo(program, get_first_device(opencl_context), CL_PROGRAM_BUILD_LOG,
                              sizeof(cBuildLog), cBuildLog, NULL );

        std::cerr << cBuildLog << std::endl;
        check_error(errNum, CL_SUCCESS);
    }

    pthread_mutex_unlock(&mtx);
    return program;
}

static void allocate_ocl_buffers(cl_context gpuContext, cl_command_queue commandQueue, const Graph &graph,
                                 cl_mem *vertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *weightArrayDevice,
                                 cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice,
                                 size_t globalWorkSize)
{
    cl_int errNum;
    cl_mem hostVertexArrayBuffer;
    cl_mem hostEdgeArrayBuffer;
    cl_mem hostWeightArrayBuffer;

    // First, need to create OpenCL Host buffers that can be copied to device buffers
    hostVertexArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph.vertex_array.size(), (void *)graph.vertex_array.data(), &errNum);
    check_error(errNum, CL_SUCCESS);

    hostEdgeArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph.edge_array.size(), (void *)graph.edge_array.data(), &errNum);
    check_error(errNum, CL_SUCCESS);

    hostWeightArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(float) * graph.edge_array.size(), (void *)graph.weight_array.data(), &errNum);
    check_error(errNum, CL_SUCCESS);

    // Now create all of the GPU buffers
    *vertexArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * globalWorkSize, NULL, &errNum);
    check_error(errNum, CL_SUCCESS);
    *edgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph.edge_array.size(), NULL, &errNum);
    check_error(errNum, CL_SUCCESS);
    *weightArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(float) * graph.edge_array.size(), NULL, &errNum);
    check_error(errNum, CL_SUCCESS);
    *maskArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    check_error(errNum, CL_SUCCESS);
    *costArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(float) * globalWorkSize, NULL, &errNum);
    check_error(errNum, CL_SUCCESS);
    *updatingCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(float) * globalWorkSize, NULL, &errNum);
    check_error(errNum, CL_SUCCESS);

    // Now queue up the data to be copied to the device
    errNum = clEnqueueCopyBuffer(commandQueue, hostVertexArrayBuffer, *vertexArrayDevice, 0, 0,
                                 sizeof(int) * graph.vertex_array.size(), 0, NULL, NULL);
    check_error(errNum, CL_SUCCESS);

    errNum = clEnqueueCopyBuffer(commandQueue, hostEdgeArrayBuffer, *edgeArrayDevice, 0, 0,
                                 sizeof(int) * graph.edge_array.size(), 0, NULL, NULL);
    check_error(errNum, CL_SUCCESS);

    errNum = clEnqueueCopyBuffer(commandQueue, hostWeightArrayBuffer, *weightArrayDevice, 0, 0,
                                 sizeof(float) * graph.edge_array.size(), 0, NULL, NULL);
    check_error(errNum, CL_SUCCESS);

    clReleaseMemObject(hostVertexArrayBuffer);
    clReleaseMemObject(hostEdgeArrayBuffer);
    clReleaseMemObject(hostWeightArrayBuffer);
}

static void init_ocl_buffers(cl_command_queue commandQueue, cl_kernel initializeKernel, const Graph &graph,
                            size_t maxWorkGroupSize)
{
    cl_int errNum;
    // Set # of work items in work group and total in 1 dimensional range
    size_t globalWorkSize [] = { 0 };
    globalWorkSize[0] = roundWorkSizeUp(maxWorkGroupSize, graph.vertex_array.size());

    errNum = clEnqueueNDRangeKernel(commandQueue, initializeKernel, 1, NULL, globalWorkSize, NULL,
                                    0, NULL, NULL);
    check_error(errNum, CL_SUCCESS);
}

cl_device_id get_first_device(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id first = cdDevices[0];
    free(cdDevices);

    return first;
}

cl_device_id get_max_flops_dev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);
    size_t device_count = szParmDataBytes / sizeof(cl_device_id);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id max_flops_device = cdDevices[0];
	int max_flops = 0;

	size_t current_device = 0;

    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);

	max_flops = compute_units * clock_frequency;
	++current_device;

	while (current_device < device_count)
	{
        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);

        int flops = compute_units * clock_frequency;
		if (flops > max_flops)
		{
			max_flops        = flops;
			max_flops_device = cdDevices[current_device];
		}
		++current_device;
	}

    free(cdDevices);

	return max_flops_device;
}

///
/// Check for error condition and exit if found.  Print file and line number
/// of error. (from NVIDIA SDK)
///
void assert_msg(int errNum, int expected, const char* file, const int lineNumber)
{
    if (errNum != expected)
    {
        std::cerr << "Assertion failed (" << errNum << " != " << expected << ") at " << file << ":" << lineNumber << std::endl;
        exit(1);
    }
}

///
/// Round the local work size up to the next multiple of the size
///
int roundWorkSizeUp(int groupSize, int globalSize)
{
    int remainder = globalSize % groupSize;
    if (remainder == 0)
    {
        return globalSize;
    }
    else
    {
        return globalSize + groupSize - remainder;
    }
}

static std::vector<float> run_dijkstra(cl_context context, cl_device_id deviceId, const Graph &graph, int source_vertex)
{
    // Create command queue
    cl_int errNum;
    cl_command_queue commandQueue;

    commandQueue = clCreateCommandQueue( context, deviceId, 0, &errNum );
    check_error(errNum, CL_SUCCESS);

    std::vector<float> shortest_path(graph.vertex_array.size(), 0);
    cl_program program = load_and_build_program(context, "dijkstra.cl");
    if (program == nullptr)
    {
        return std::vector<float>();
    }

    // Get the max workgroup size
    size_t maxWorkGroupSize = 0;
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);


    // Set # of work items in work group and total in 1 dimensional range
    size_t globalWorkSize = roundWorkSizeUp(maxWorkGroupSize, graph.vertex_array.size());

    cl_mem vertexArrayDevice;
    cl_mem edgeArrayDevice;
    cl_mem weightArrayDevice;
    cl_mem maskArrayDevice;
    cl_mem costArrayDevice;
    cl_mem updatingCostArrayDevice;

    // Allocate buffers in Device memory
    allocate_ocl_buffers(context, commandQueue, graph, &vertexArrayDevice, &edgeArrayDevice, &weightArrayDevice,
                         &maskArrayDevice, &costArrayDevice, &updatingCostArrayDevice, globalWorkSize);


    // Create the Kernels
    cl_kernel initializeBuffersKernel;
    initializeBuffersKernel = clCreateKernel(program, "initializeBuffers", &errNum);
    check_error(errNum, CL_SUCCESS);

    // Set the args values and check for errors
    errNum |= clSetKernelArg(initializeBuffersKernel, 0, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 1, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 2, sizeof(cl_mem), &updatingCostArrayDevice);

    // 3 set below in loop
    auto vertex_count = graph.vertex_array.size();
    auto edge_count = graph.edge_array.size();
    errNum |= clSetKernelArg(initializeBuffersKernel, 4, sizeof(int), &vertex_count);
    check_error(errNum, CL_SUCCESS);

    // Kernel 1
    cl_kernel ssspKernel1;
    ssspKernel1 = clCreateKernel(program, "OCL_SSSP_KERNEL1", &errNum);
    check_error(errNum, CL_SUCCESS);
    errNum |= clSetKernelArg(ssspKernel1, 0, sizeof(cl_mem), &vertexArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 1, sizeof(cl_mem), &edgeArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 2, sizeof(cl_mem), &weightArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 3, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 4, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 5, sizeof(cl_mem), &updatingCostArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 6, sizeof(int), &vertex_count);
    errNum |= clSetKernelArg(ssspKernel1, 7, sizeof(int), &edge_count);
    check_error(errNum, CL_SUCCESS);

    // Kernel 2
    cl_kernel ssspKernel2;
    ssspKernel2 = clCreateKernel(program, "OCL_SSSP_KERNEL2", &errNum);
    check_error(errNum, CL_SUCCESS);
    errNum |= clSetKernelArg(ssspKernel2, 0, sizeof(cl_mem), &vertexArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 1, sizeof(cl_mem), &edgeArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 2, sizeof(cl_mem), &weightArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 3, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 4, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 5, sizeof(cl_mem), &updatingCostArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 6, sizeof(int), &vertex_count);

    check_error(errNum, CL_SUCCESS);

    int *maskArrayHost = new int[graph.vertex_array.size()];

    errNum |= clSetKernelArg(initializeBuffersKernel, 3, sizeof(int), &source_vertex);
    check_error(errNum, CL_SUCCESS);

    // Initialize mask array to false, C and U to infiniti
    init_ocl_buffers(commandQueue, initializeBuffersKernel, graph, maxWorkGroupSize);

    // Read mask array from device -> host
    cl_event readDone;
    errNum = clEnqueueReadBuffer(commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * graph.vertex_array.size(),
                                 maskArrayHost, 0, NULL, &readDone);
    check_error(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);

    while (!maskArrayEmpty(maskArrayHost, graph.vertex_array.size()))
    {
        // In order to improve performance, we run some number of iterations
        // without reading the results.  This might result in running more iterations
        // than necessary at times, but it will in most cases be faster because
        // we are doing less stalling of the GPU waiting for results.
        for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
        {
            size_t globalWorkSize = roundWorkSizeUp(maxWorkGroupSize, graph.vertex_array.size());

            // execute the kernel
            errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel1, 1, 0, &globalWorkSize, NULL,
                                            0, NULL, NULL);
            check_error(errNum, CL_SUCCESS);

            errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel2, 1, 0, &globalWorkSize, NULL,
                                            0, NULL, NULL);
            check_error(errNum, CL_SUCCESS);
        }
        errNum = clEnqueueReadBuffer(commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * graph.vertex_array.size(),
                                        maskArrayHost, 0, NULL, &readDone);
        check_error(errNum, CL_SUCCESS);
        clWaitForEvents(1, &readDone);
    }

    // Copy the result back
    errNum = clEnqueueReadBuffer(commandQueue, costArrayDevice, CL_FALSE, 0, sizeof(float) * graph.vertex_array.size(),
                                 shortest_path.data(), 0, NULL, &readDone);
    check_error(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);


    delete[] maskArrayHost;

    clReleaseMemObject(vertexArrayDevice);
    clReleaseMemObject(edgeArrayDevice);
    clReleaseMemObject(weightArrayDevice);
    clReleaseMemObject(maskArrayDevice);
    clReleaseMemObject(costArrayDevice);
    clReleaseMemObject(updatingCostArrayDevice);

    clReleaseKernel(initializeBuffersKernel);
    clReleaseKernel(ssspKernel1);
    clReleaseKernel(ssspKernel2);

    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(program);

    return shortest_path;
}


///
/// Check whether the mask array is empty.  This tells the algorithm whether
/// it needs to continue running or not.
///
bool maskArrayEmpty(int *maskArray, int count)
{
    for(int i = 0; i < count; i++ )
    {
        if (maskArray[i] == 1)
        {
            return false;
        }
    }

    return true;
}

ocl_init_result_t dijkstra_init_contexts(cl_context &gpu_context, cl_context &cpu_context)
{
    cl_platform_id platform;
    cl_int err_num;

    bool cpu_found = true;
    bool gpu_found = true;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    cl_uint numPlatforms;
    err_num = clGetPlatformIDs(1, &platform, &numPlatforms);

    std::cout << "Number of OpenCL Platforms: " << numPlatforms << std::endl;
    if (err_num != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cout << "Failed to find any OpenCL platforms" << std::endl;
        return OCL_INIT_NO_PLATFORM;
    }
    print_device_info();
    // create the OpenCL context on available GPU devices
    gpu_context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        std::cout << "No GPU devices found." << std::endl;
        gpu_found = false;
    }

    // Create an OpenCL context on available CPU devices
    cpu_context = clCreateContextFromType(0, CL_DEVICE_TYPE_CPU, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        std::cout << "No CPU devices found." << std::endl;
        cpu_found = false;
    }

    if (!gpu_found && !cpu_found)
    {
        return OCL_INIT_NO_DEVICES;
    }

    if (cpu_found && !gpu_found)
    {
        return OCL_INIT_CPU_ONLY;
    }

    if (!cpu_found && gpu_found)
    {
        return OCL_INIT_GPU_ONLY;
    }

    return OCL_INIT_SUCCESS;
}

std::vector<float> dijkstra_opencl(const Graph &graph, int source_vertex, cl_context &opencl_context)
{
    return run_dijkstra(opencl_context, get_max_flops_dev(opencl_context), graph, source_vertex);
}
