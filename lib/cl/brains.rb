require "cl/brains/version"
require 'opencl_ruby_ffi'
require 'narray_ffi'



module Cl
  module Brains
    class CLMatrixMath
      def initialize(cols, device = nil)
        @cols = cols
        device ||= begin
          platform = OpenCL::platforms.first
          platform.devices.first
        end
        @context = OpenCL::create_context(device)
        @queue = @context.create_command_queue(device, :properties => OpenCL::CommandQueue::PROFILING_ENABLE)
        matrix_source = <<EOF
  __kernel void matrix(uint col_size, __global const int *A, __global const int *B, __global int *C) {

    // Get the index of the current element to be processed
    int i = get_global_id(0) / col_size;
    int j = get_global_id(0) % col_size;
    
        
    int value = 0;
    for(int k = 0; k < col_size; k++)
    {
        value += A[i * col_size + k] *  B[k * col_size + j];
    }
    C[i * col_size + j] = value;
}
EOF

        @prog = @context.create_program_with_source( matrix_source )
        @prog.build
      end

      def matrix_mul_int(a, b)
        matrix_a = NArray.int(@cols * @cols)
        matrix_b = NArray.int(@cols * @cols)
        matrix_c = NArray.int(@cols * @cols)
        a.flatten.each_with_index do |element, index|
          matrix_a[index] = element
        end

        b.flatten.each_with_index do |element, index|
          matrix_b[index] = element
        end

        b_matrix_a = @context.create_buffer(matrix_a.size * matrix_a.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => matrix_a)
        b_matrix_b = @context.create_buffer(matrix_b.size * matrix_b.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => matrix_b)
        b_matrix_c = @context.create_buffer(matrix_c.size * matrix_c.element_size)
        f = OpenCL::Int1::new(@cols)
        event = @prog.matrix(@queue, [@cols * @cols], f, b_matrix_a, b_matrix_b, b_matrix_c)
        @queue.enqueue_read_buffer(b_matrix_c, matrix_c, :event_wait_list => [event])
        @queue.finish
        matrix_c.to_a.each_slice(@cols).collect { |slice| slice }
      end
    end
  end
end
