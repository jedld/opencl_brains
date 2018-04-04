require 'ostruct'

module Cl
  module Brains
    class Tensor
      attr_accessor :data_type, :shape, :rank, :native_buffer

      def initialize(data_type, rank, shape)
        @data_type = data_type
        @rank = rank
        @shape = OpenStruct.new(shape)

        @native_buffer = if data_type == :float
          NArray.sfloat(@shape.cols * @shape.rows)
        else
          raise "Invalid data type #{data_type}"
        end
      end

      def self.matrix(m)
        cols = m[0].size
        rows = m.size

        tensor = Cl::Brains::Tensor.new(:float, 2, {cols: cols, rows: rows} )
        
        m.flatten.each_with_index do |element, index|
          tensor.native_buffer[index] = element
        end

        tensor
      end

      def to_ary
        if rank == 2
          @native_buffer.to_a.each_slice(shape.cols).collect { |slice| slice }
        else
          raise "Invalid rank"
        end
      end

      # open cl methods
      def open_cl_buffer(context)
        @cl_buffer ||= context.create_buffer(@native_buffer.size * @native_buffer.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => @native_buffer)
      end

      def sync_cl_buffer(queue, events = [])
        queue.enqueue_read_buffer(@cl_buffer, @native_buffer, :event_wait_list => events)
      end
    end
  end
end
