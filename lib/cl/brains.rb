require "cl/brains/version"
require 'opencl_ruby_ffi'
require 'narray_ffi'
require 'cl/brains/tensor'
require "cl/brains/matrix_operation"
require "cl/brains/sigmoid_operation"

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
end
