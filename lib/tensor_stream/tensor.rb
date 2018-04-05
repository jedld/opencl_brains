require 'ostruct'

module TensorStream
  class Tensor
    attr_accessor :name, :data_type, :shape, :rank, :native_buffer, :is_const

    def self.const_name
      @@const_counter ||= 0

      name = if @@const_counter == 0
        ""
      else
        "_#{@@const_counter}"
      end

      @@const_counter += 1
      
      name
    end

    def self.var_name
      @@var_counter ||= 0
      @@var_counter += 1

      return "" if @@var_counter == 1
      return "_#{@@var_counter}"
    end

    def initialize(data_type, rank, shape, options = {})
      @data_type = data_type
      @rank = rank
      @shape = TensorShape.new(shape, rank)

      @is_const = options[:const] || false
      @name = "#{is_const ? "Const#{Tensor.const_name}:#{rank}" : "Variable#{Tensor.var_name}:#{rank}"}"
    end

    def self.Variable(value, dtype = nil)
      if value.is_a?(String)
        TensorStream::Tensor.new(dtype || :string_ref, 0, [])
      elsif value.is_a?(Integer)
        TensorStream::Tensor.new(dtype || :int32_ref, 0, [])
      elsif value.is_a?(Float)
        TensorStream::Tensor.new(dtype || :float32_ref, 0, [])
      end
    end

    def self.constant(value, options = {})
      if value.is_a?(Float)
        TensorStream::Tensor.new(options[:dtype] || :float32, 0, [], { const: true })
      elsif value.is_a?(Integer)
        TensorStream::Tensor.new(options[:dtype] || :int32, 0, [], { const: true })
      elsif value.is_a?(String)
        TensorStream::Tensor.new(options[:dtype] || :string, 0, [], { const: true })
      elsif value.is_a?(Array)
        dtype = nil
        rank = 1
        dimensions = []
        begin
          dtype, rank, value, d = dtype_eval(dtype, rank, value)
          dimensions << d
        end while dtype == :array
 
        TensorStream::Tensor.new(dtype, rank, dimensions, { const: true })
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

    def build_buffer
      @native_buffer = if @data_type == :float32 && @rank == 2
        NArray.sfloat(@shape.cols * @shape.rows)
      elsif @data_type == :float32 && @rank == 0
        NArray.sfloat(1)
      else
        raise "Invalid data type #{@data_type}"
      end
    end

    def +(operand)
      TensorStream::Operation.new(:add, self, operand)
    end

    def to_s
      @name
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

    protected

    def self.dtype_eval(dtype, rank, value)
      dtype = if value[0].is_a?(String)
        :string
      elsif value[0].is_a?(Float)
        :float32
      elsif value[0].is_a?(Integer)
        :int32
      elsif value[0].is_a?(Array)
        rank += 1
        :array
      else
        :float32
      end

      [dtype, rank, value[0], value.size]
    end
  end
end
