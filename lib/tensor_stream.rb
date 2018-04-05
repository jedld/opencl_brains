require "tensor_stream/version"
require 'opencl_ruby_ffi'
require 'narray_ffi'
require 'tensor_stream/types'
require 'tensor_stream/tensor_shape'
require 'tensor_stream/tensor'
require 'tensor_stream/operation'
require "tensor_stream/gemm/gemm"
require "tensor_stream/sigmoid/sigmoid"

module TensorStream
  class Session
    def initialize(cols, device = nil)
      @cols = cols
      device ||= begin
        platform = OpenCL::platforms.first
        platform.devices.first
      end
      @context = OpenCL::create_context(device)
      @queue = @context.create_command_queue(device, :properties => OpenCL::CommandQueue::PROFILING_ENABLE)
    end

    def run(operations)
    end

    def mul(matrix_a, matrix_b)
      matrix_op = Cl::Brains::MatrixOperation.new(@context, @prog, @queue)
      matrix_op.program!
      matrix_op.nativize(matrix_a: matrix_a, matrix_b: matrix_b)
      matrix_op
    end

    def sigmoid(matrix_a)
      matrix_op = Cl::Brains::SigmoidOperation.new(@context, @prog, @queue)
      matrix_op.program!
      matrix_op.nativize(matrix_a: matrix_a)
      matrix_op
    end
  end
end
