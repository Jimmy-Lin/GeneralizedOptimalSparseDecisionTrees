#include "index.hpp"

#ifdef INCLUDE_OPENCL
cl::Device Index::device = cl::Device();
cl::Platform Index::platform = cl::Platform();
cl::Context Index::context = cl::Context();
cl::Program Index::program = cl::Program();
cl::CommandQueue Index::queue = cl::CommandQueue();
unsigned int Index::group_size = 48;
#endif

Index::Index(void) {}

Index::Index(std::vector< std::vector< float > > const & src) {
    this -> size = src.size();
    this -> width = src.begin() -> size();
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, &number_of_blocks, &block_offset);
    this -> num_blocks = number_of_blocks;

    // this -> parallel_threshold = ~0; // Use max int to disable parallel for now

    build_prefixes(src, this -> prefixes);

    this -> source.resize(this -> size * this -> width, 0.0);
    for (unsigned int i = 0;  i < this -> size; ++i) {
        for (unsigned int j = 0; j < this -> width; ++j) {
            this -> source[i * this -> width + j] = src.at(i).at(j);
        }
    }
    initialize_kernel();
    // benchmark();
}

Index::~Index(void) {}


void Index::precompute(void) {
}

void Index::build_prefixes(std::vector< std::vector< float > > const & src, std::vector< std::vector< float > > & prefixes) {
    // Compute a scan over the source vector, tracker[i] == sum(source[0..i-1])
    std::vector< float > init(this -> width, 0.0); // First entries for the prefix sum
    prefixes.emplace_back(init);
    for (unsigned int i = 0; i < this -> size; i++) {
        std::vector< float > const & source_row = src.at(i);
        std::vector< float > const & prefix_row = prefixes.at(i);
        std::vector< float > new_row;
        for (unsigned int j = 0; j < this -> width; ++j) {
            new_row.emplace_back(source_row.at(j) + prefix_row.at(j));
        }
        prefixes.emplace_back(new_row);
    }
}

void Index::sum(Bitmask const & indicator, float * accumulator) const {
    bit_sequential_sum(indicator, accumulator);
    float const epsilon = std::numeric_limits<float>::epsilon();
    for (unsigned int j = 0; j < this -> width; ++j) {
        if (accumulator[j] < epsilon) { accumulator[j] = 0.0; }
    }
}

void Index::bit_sequential_sum(Bitmask const & indicator, float * accumulator) const {
    unsigned int max = indicator.size();
    bool sign = true;
    unsigned int i = indicator.scan(0, true);;
    unsigned int j = indicator.scan(i, !sign);
    while (j <= max) {
        if (sign) {
            {
                std::vector< float > const & finish = this -> prefixes.at(j);
                for (int k = this -> width; --k >= 0;) { accumulator[k] += finish.at(k); }
            }
            {
                std::vector< float > const & start = this -> prefixes.at(i);
                for (int k = this -> width; --k >= 0;) { accumulator[k] -= start.at(k); }
            }
        }
        if (j == max) { break; }
        i = j;
        sign = !sign;
        j = indicator.scan(i, !sign);
    }
}

void Index::block_sequential_sum(bitblock * blocks, float * accumulator) const {
    unsigned int offset = 0;
    for (unsigned int i = 0; i < this -> num_blocks; ++i) {
        bitblock block = blocks[i];
        for (unsigned int range_index = 0; range_index < sizeof(bitblock) / sizeof(rangeblock); ++range_index) {
            unsigned int local_offset = offset + range_index * 8 * sizeof(rangeblock);
            unsigned int shift = range_index * 8 * sizeof(rangeblock);
            block_sequential_sum(0xffff & (block >> shift), local_offset, accumulator);
        }
        offset += 8 * sizeof(bitblock);
    }
}

void Index::block_sequential_sum(rangeblock block, unsigned int offset, float * accumulator) const {
    unsigned int local_offset = offset;
    bool positive = ((block) & 1) == 1;

    std::vector<codeblock> const & encoding = Bitmask::ranges[block];
    for (auto iterator = encoding.begin(); iterator != encoding.end(); ++iterator) {
        codeblock packed_code = * iterator;
        unsigned short code;

        for (unsigned int range_index = 0; range_index < Bitmask::ranges_per_code; ++range_index) {
            if (local_offset >= offset + 16 || local_offset >= this -> size) { break; }

            unsigned int shift = range_index * Bitmask::bits_per_range;
            code = (0xF & (packed_code >> shift)) + 1;

            if (positive) {
                std::vector< float > const & start = this -> prefixes.at(local_offset);
                std::vector< float > const & finish = this -> prefixes.at(local_offset + code);
                for (unsigned int j = 0; j < this -> width; ++j) { accumulator[j] += finish.at(j) - start.at(j); }
            }
            local_offset += code;
            positive = !positive;
        }
    }
}

std::string Index::to_string(void) const {
    std::stringstream stream;
    stream << "[";
    for (unsigned int i = 0; i < this -> size; ++i) {
        for (unsigned int j = 0; j < this -> width; ++j) {
            stream << this -> source[i * this -> width + j];
            stream << ",";
        }
        if (i+1 < this -> size) { stream << std::endl; }
    }
    stream << "]";
    return stream.str();
}


void Index::initialize_kernel(void) {
#ifdef INCLUDE_OPENCL
    set_platform(0);
    set_device(Index::platform, 1);

    std::cout << "Using platform: " << Index::platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Using device: " << Index::device.getInfo<CL_DEVICE_NAME>() << std::endl;

    Index::context = cl::Context({Index::device});
    cl::Program::Sources sources;
    std::stringstream kernel_stream;
    kernel_stream << "void kernel selective_sum(global const unsigned int * selector, global const float * data, global float * output) {" << std::endl;
    kernel_stream << "   unsigned int global_id = get_global_id(0);" << std::endl;
    kernel_stream << "   unsigned int group_id = global_id / " << Index::group_size << ";" << std::endl;
    kernel_stream << "   unsigned int local_id = global_id % " << Index::group_size << ";" << std::endl;

    kernel_stream << "   unsigned int block = selector[global_id / " << 8 * Bitmask::bits_per_block << "];" << std::endl;
    kernel_stream << "   float selected = (block >> (global_id % " << 8 * Bitmask::bits_per_block << ")) & 1;" << std::endl;

    kernel_stream << "   local float working[" << (Index::group_size * this -> width) << "];" << std::endl;
    kernel_stream << "   for (unsigned int col = 0; col < " << this -> width << "; ++col) {" << std::endl;
    kernel_stream << "       if (global_id < " << this -> size << ") {" << std::endl;
    kernel_stream << "           working[local_id * " << this -> width << " + col] = selected * data[global_id * " << this -> width << " + col];" << std::endl;
    kernel_stream << "       } else {" << std::endl;
    kernel_stream << "           working[local_id * " << this -> width << " + col] = 0.0;" << std::endl;
    kernel_stream << "       }" << std::endl;
    kernel_stream << "   }" << std::endl;
    kernel_stream << "   barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

    kernel_stream << "   unsigned int size = " << this -> size << ";" << std::endl;
    kernel_stream << "   unsigned int groups = " << this -> num_blocks << ";" << std::endl;
    kernel_stream << "   while (size > 1) {" << std::endl;

    kernel_stream << "       if (group_id <= groups) {" << std::endl;
    kernel_stream << "           for (unsigned int stride = " << Index::group_size / 2 << "; stride >= 1; stride /= 2) {" << std::endl;
    kernel_stream << "               if (local_id < stride) {" << std::endl;
    kernel_stream << "                   for (unsigned int col = 0; col < " << this -> width << "; ++col) {" << std::endl;
    kernel_stream << "                       working[local_id * " << this -> width << " + col] += working[(local_id + stride) * " << this -> width << " + col];" << std::endl;
    kernel_stream << "                   }" << std::endl;
    kernel_stream << "               }" << std::endl;
    kernel_stream << "               barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
    kernel_stream << "               if (stride == 1) { break; }" << std::endl;
    kernel_stream << "           }" << std::endl;
    kernel_stream << "       }" << std::endl;

    kernel_stream << "       if (local_id == 0) {" << std::endl;
    kernel_stream << "           for (unsigned int col = 0; col < " << this -> width << "; ++col) {" << std::endl;
    kernel_stream << "               output[group_id * " << this -> width << " + col] = working[local_id * " << this -> width << " + col];" << std::endl;
    kernel_stream << "           }" << std::endl;
    kernel_stream << "       }" << std::endl;
    kernel_stream << "       barrier(CLK_GLOBAL_MEM_FENCE);" << std::endl;

    kernel_stream << "       for (unsigned int col = 0; col < " << this -> width << "; ++col) {" << std::endl;
    kernel_stream << "           working[local_id * " << this -> width << " + col] = 0.0;" << std::endl;
    kernel_stream << "       }" << std::endl;

    kernel_stream << "       if (global_id < size) {" << std::endl;
    kernel_stream << "           for (unsigned int col = 0; col < " << this -> width << "; ++col) {" << std::endl;
    kernel_stream << "               working[local_id * " << this -> width << " + col] = output[global_id * " << this -> width << " + col];" << std::endl;
    kernel_stream << "           }" << std::endl;
    kernel_stream << "       }" << std::endl;
    kernel_stream << "       barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

    kernel_stream << "       size = groups;" << std::endl;
    kernel_stream << "       groups = size / " << Index::group_size << " + (unsigned int)(size % groups == 0);" << std::endl;
    kernel_stream << "   }" << std::endl;

    kernel_stream << "   return;" << std::endl;
    kernel_stream << "}" << std::endl;

    std::string kernel_source = kernel_stream.str();
    sources.push_back({kernel_source.c_str(), kernel_source.length()});
    Index::program = cl::Program(Index::context, sources);
    if (Index::program.build({Index::device}) != CL_SUCCESS) {
        std::cout << "Error building: " << Index::program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(Index::device) << std::endl;
        exit(1);
    }

    Index::queue = cl::CommandQueue(Index::context, Index::device, CL_QUEUE_PROFILING_ENABLE);
    // Index::queue = cl::CommandQueue(Index::context, Index::device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    std::vector< float > data = this -> source;
    this -> data_buffer = cl::Buffer(Index::context, CL_MEM_READ_ONLY, sizeof(float) * data.size());
    int error_code = Index::queue.enqueueWriteBuffer(data_buffer, CL_TRUE, 0, sizeof(float) * data.size(), data.data());
    if (error_code != CL_SUCCESS) {
        std::cout << "Error writing data input" << std::endl;
        std::cout << "Error Code: " << error_code << std::endl;
        exit(1);
    }
#endif
}

#ifdef INCLUDE_OPENCL
void Index::set_platform(int index) {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << "No platforms found. Check OpenCL installation!\n" << std::endl;
        exit(1);
    }
    Index::platform = all_platforms[index];
}

void Index::set_device(cl::Platform platform, int index, bool display) {
    std::vector<cl::Device> all_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << "No devices found. Check OpenCL installation!\n" << std::endl;
        exit(1);
    }

    if (display) {
        for (int j = 0; j < all_devices.size(); j++)
            printf("Device %d: %s\n", j, all_devices[j].getInfo<CL_DEVICE_NAME>().c_str());
    }
    Index::device = all_devices[index];
}

void Index::parallel_sum(bitblock * blocks, float * accumulator, bool blocking, bool profile) const {
    unsigned int group_count = (this -> size / Index::group_size) + (this -> size % Index::group_size != 0);
    unsigned int thread_count = group_count * Index::group_size;

    std::vector< float > output((this -> width) * (group_count), 0.0);

    cl::Buffer select_buffer(Index::context, CL_MEM_READ_ONLY, sizeof(unsigned int) * this -> num_blocks);
    cl::Buffer output_buffer(Index::context, CL_MEM_READ_WRITE, sizeof(float) * output.size());

    int error_code;

    cl::Event input_event;
    error_code = Index::queue.enqueueWriteBuffer(select_buffer, CL_FALSE, 0, sizeof(unsigned int) * this -> num_blocks, blocks,
         NULL, & input_event);
    if (error_code != CL_SUCCESS) {
        std::cout << "Error writing selector input" << std::endl;
        std::cout << "Error Code: " << error_code << std::endl;
        exit(1);
    }

    cl::Kernel kernel(Index::program, "selective_sum");
    error_code = kernel.setArg(0, select_buffer);
    if (error_code != CL_SUCCESS) {
        std::cout << "Error setting argument 0" << std::endl;
        std::cout << "Error Code: " << error_code << std::endl;
        exit(1);
    }
    error_code = kernel.setArg(1, this -> data_buffer);
    if (error_code != CL_SUCCESS) {
        std::cout << "Error setting argument 1" << std::endl;
        std::cout << "Error Code: " << error_code << std::endl;
        exit(1);
    }
    error_code = kernel.setArg(2, output_buffer);
    if (error_code != CL_SUCCESS) {
        std::cout << "Error setting argument 2" << std::endl;
        std::cout << "Error Code: " << error_code << std::endl;
        exit(1);
    }

    cl::Event execution_event;
    std::vector<cl::Event> execution_dependencies{input_event};
    error_code = Index::queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(thread_count), cl::NDRange(Index::group_size),
        & execution_dependencies, & execution_event);
    if (error_code != CL_SUCCESS) {
        std::cout << "Error enqueing kernel" << std::endl;
        std::cout << "Error Code: " << error_code << std::endl;
        exit(1);
    }

    cl::Event output_event;
    std::vector<cl::Event> output_dependencies{execution_event};
    // error_code = Index::queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float) * output.size(), output.data(), NULL, & output_event);
    error_code = Index::queue.enqueueReadBuffer(output_buffer, CL_FALSE, 0, sizeof(float) * this -> width, accumulator,
        & output_dependencies, & output_event);
    if (error_code != CL_SUCCESS) {
        std::cout << "Error reading output" << std::endl;
        std::cout << "Error Code: " << error_code << std::endl;
        exit(1);
    };

    if (blocking) { output_event.wait(); }

    // for (unsigned int k = 0; k < group_count; ++k) {
    //     for (unsigned int j = 0; j < this -> width; ++j) {
    //         accumulator[j] += output.at(k * this -> width + j);
    //     }
    // }

    output_event.wait();

    if (profile) {
        unsigned long queued, submitted, started, finished;

        input_event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, & queued);
        input_event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, & submitted);
        input_event.getProfilingInfo(CL_PROFILING_COMMAND_START, & started);
        input_event.getProfilingInfo(CL_PROFILING_COMMAND_END, & finished);
        std::cout << "GPU Input Profile:" << std::endl;
        std::cout << "  Total: " << (finished - queued) << " ns" << std::endl;
        std::cout << "  Queue: " << (submitted - queued) << " ns" << std::endl;
        std::cout << "  Transmission: " << (started - submitted) << " ns" << std::endl;
        std::cout << "  Execution: " << (finished - started) << std::endl;

        execution_event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued);
        execution_event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &submitted);
        execution_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &started);
        execution_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &finished);
        std::cout << "GPU Execution Profile:" << std::endl;
        std::cout << "  Total: " << (finished - queued) << " ns" << std::endl;
        std::cout << "  Queue: " << (submitted - queued) << " ns" << std::endl;
        std::cout << "  Transmission: " << (started - submitted) << " ns" << std::endl;
        std::cout << "  Execution: " << (finished - started) << std::endl;

        output_event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued);
        output_event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &submitted);
        output_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &started);
        output_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &finished);
        std::cout << "GPU Output Profile:" << std::endl;
        std::cout << "  Total: " << (finished - queued) << " ns" << std::endl;
        std::cout << "  Queue: " << (submitted - queued) << " ns" << std::endl;
        std::cout << "  Transmission: " << (started - submitted) << " ns" << std::endl;
        std::cout << "  Execution: " << (finished - started) << std::endl;

        exit(1);
    }
}
#endif

void Index::benchmark(void) const
{
    Bitmask indicator(this->size, true);
    for (unsigned int i = 0; i < this->size; ++i)
    {
        indicator.set(i, (i % 7 != 0));
    }
    bitblock *blocks = indicator.data();
    std::vector<float, tbb::scalable_allocator<float>> accumulator(this->width);
    unsigned int sample_size = 10000;

    float block_min = std::numeric_limits<float>::max();
    float block_max = -std::numeric_limits<float>::max();
    float block_avg;

    float bit_min = std::numeric_limits<float>::max();
    float bit_max = -std::numeric_limits<float>::max();
    float bit_avg;

#ifdef INCLUDE_OPENCL
    float par_min = std::numeric_limits<float>::max();
    float par_max = -std::numeric_limits<float>::max();
    float par_avg;
#endif

    auto block_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        block_sequential_sum(blocks, accumulator.data());
        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);

        block_min = std::min((float)duration.count() / 1000, block_min);
        block_max = std::max((float)duration.count() / 1000, block_max);
    }
    auto block_finish = std::chrono::high_resolution_clock::now();
    block_avg = (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(block_finish - block_start).count()) / sample_size / 1000;

    auto bit_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        bit_sequential_sum(indicator, accumulator.data());
        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);

        bit_min = std::min((float)duration.count() / 1000, bit_min);
        bit_max = std::max((float)duration.count() / 1000, bit_max);
    }
    auto bit_finish = std::chrono::high_resolution_clock::now();
    bit_avg = (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(bit_finish - bit_start).count()) / sample_size / 1000;

#ifdef INCLUDE_OPENCL
    auto par_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        parallel_sum(blocks, accumulator.data(), true, false);
        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);

        par_min = std::min((float)duration.count() / 1000, par_min);
        par_max = std::max((float)duration.count() / 1000, par_max);
    }
    auto par_finish = std::chrono::high_resolution_clock::now();
    par_avg = (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(par_finish - par_start).count()) / sample_size / 1000;
#endif

    std::cout << "Index Benchmark Results: " << std::endl;
    std::cout << "Block Sequential: " << std::endl;
    std::cout << "  Min: " << block_min << " ms" << std::endl;
    std::cout << "  Avg: " << block_avg << " ms" << std::endl;
    std::cout << "  Max: " << block_max << " ms" << std::endl;
    std::cout << "Bit Sequential: " << std::endl;
    std::cout << "  Min: " << bit_min << " ms" << std::endl;
    std::cout << "  Avg: " << bit_avg << " ms" << std::endl;
    std::cout << "  Max: " << bit_max << " ms" << std::endl;

#ifdef INCLUDE_OPENCL
    std::cout << "Parallel: " << std::endl;
    std::cout << "  Min: " << par_min << " ms" << std::endl;
    std::cout << "  Avg: " << par_avg << " ms" << std::endl;
    std::cout << "  Max: " << par_max << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "GPU Profile Results: " << std::endl;
    parallel_sum(blocks, accumulator, true, true);
#endif

#ifdef INCLUDE_OPENCL
    // Match results
    std::vector<float, tbb::scalable_allocator<float>> block_accumulator(this->width);
    std::vector<float, tbb::scalable_allocator<float>> bit_accumulator(this->width);
    std::vector<float, tbb::scalable_allocator<float>> par_accumulator(this->width);
    bit_sequential_sum(blocks, bit_accumulator);
    block_sequential_sum(blocks, block_accumulator);
    parallel_sum(blocks, par_accumulator, true, false);

    bool mismatch;
    for (unsigned int j = 0; j < this->width; ++j)
    {
        if (std::abs(bit_accumulator.at(j) - par_accumulator.at(j)) > 10 * std::numeric_limits<float>::epsilon())
        {
            mismatch = true;
        }
    }
    if (mismatch)
    {
        std::cout << "Mismatch Detected for Indicator: " << indicator.to_string() << std::endl;
        std::cout << "Expected: ";
        for (unsigned int j = 0; j < this->width; ++j)
        {
            std::cout << bit_accumulator.at(j) << ",";
        }
        std::cout << std::endl;
        std::cout << "Got: ";
        for (unsigned int j = 0; j < this->width; ++j)
        {
            std::cout << par_accumulator.at(j) << ",";
        }
        std::cout << std::endl;
    }
#endif
    exit(1);
}
