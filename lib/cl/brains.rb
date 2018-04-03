require "cl/brains/version"
require 'opencl_ruby_ffi'
require 'narray_ffi'
require "cl/brains/matrix_operation"



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
        matrix_source = File.read(File.join(File.dirname(__FILE__), 'brains', 'matrix.cl'))

        @prog = @context.create_program_with_source( matrix_source )
        @prog.build
      end

      def prepare(matrix_a, matrix_b)
        Cl::Brains::MatrixOperation.new(@context, @prog, @queue, matrix_a, matrix_b)
      end
    end
  end
end
