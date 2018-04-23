require 'ostruct'

module TensorStream
  class Tensor
    attr_accessor :name, :data_type, :shape, :rank, :native_buffer, :is_const, :value

    def self.const_name
      @const_counter ||= 0

      name = if @const_counter == 0
        ""
      else
        "_#{@const_counter}"
      end

      @const_counter += 1
      
      name
    end

    def self.var_name
      @var_counter ||= 0
      @var_counter += 1

      return "" if @var_counter == 1
      return "_#{@var_counter}"
    end

    def self.placeholder_name
      @placeholder_counter ||= 0
      @placeholder_counter += 1

      return "" if @placeholder_counter == 1
      return "_#{@placeholder_counter}"
    end

    def initialize(data_type, rank, shape, options = {})
      @data_type = data_type
      @rank = rank
      @shape = TensorShape.new(shape, rank)
      @value = nil
      @is_const = options[:const] || false
      @graph = options[:graph] || TensorStream.get_default_graph
      @name = options[:name] || build_name
      if options[:value]
        if options[:value].kind_of?(Array)
          # check if single dimenstion array is passed
          if shape.size >= 2 && options[:value].size > 0 && !options[:value][0].kind_of?(Array)
            options[:value] = reshape(options[:value], shape.reverse.dup)
          end

          @value = options[:value].collect do |v|
            v.kind_of?(Tensor) ? Tensor.cast_dtype(v, data_type) : TensorStream.constant(Tensor.cast_dtype(v, data_type), dtype: data_type)
          end
        elsif shape.size > 0
          @value = reshape(options[:value], shape.reverse.dup)
        else
          @value = Tensor.cast_dtype(options[:value], @data_type)
        end
      end

      @graph.add_node(self)
    end

    def dtype
      @data_type
    end

    def self.reset_counters
      @const_counter = 0
      @var_counter = 0
      @placeholder_counter = 0
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
      TensorStream::Operation.new(:add, self, auto_wrap(operand))
    end

    def [](index)
      TensorStream::Operation.new(:slice, self, index)
    end

    def *(operand)
      TensorStream::Operation.new(:mul, self, auto_wrap(operand))
    end

    def **(operand)
      TensorStream::Operation.new(:pow, self, auto_wrap(operand))
    end

    def /(operand)
      TensorStream::Operation.new(:div, self, auto_wrap(operand))
    end

    def -(operand)
      TensorStream::Operation.new(:sub, self, auto_wrap(operand))
    end

    def -@
      TensorStream::Operation.new(:negate, self, nil)
    end

    def collect(&block)
      @value.collect(&block)
    end

    def to_s
      @name
    end

    # def to_ary
    #   if rank == 2
    #     @native_buffer.to_a.each_slice(shape.cols).collect { |slice| slice }
    #   else
    #     raise "Invalid rank"
    #   end
    # end

    # open cl methods
    def open_cl_buffer(context)
      @cl_buffer ||= context.create_buffer(@native_buffer.size * @native_buffer.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => @native_buffer)
    end

    def sync_cl_buffer(queue, events = [])
      queue.enqueue_read_buffer(@cl_buffer, @native_buffer, :event_wait_list => events)
    end

    def eval(options = {})
      Session.default_session.run(self, options)
    end

    def to_h
      {
        name: @name,
        value: hashify_tensor(@value),
        dtype: @data_type,
        shape: @shape,
        const: !!is_const,
      }
    end

    def to_i
      @value
    end

    def to_a
      @value
    end

    def to_f
      @value
    end

    def to_math
      if @value.kind_of?(Array)
        @value.collect { |v| v.kind_of?(Tensor) ? v.to_math : v }
      else
        is_const ? @value : @name
      end
    end

    def auto_math(tensor)
      tensor.kind_of?(Tensor) ? tensor.to_math : tensor
    end

    def self.detect_type(value)
      dtype = if value.is_a?(String)
        :string
      elsif value.is_a?(Float)
        :float32
      elsif value.is_a?(Integer)
        :int32
      elsif value.is_a?(Array)
        :array
      else
        :float32
      end
    end

    def self.cast_dtype(val, dtype)
      return val if val.kind_of?(Tensor)

      if val.kind_of?(Array)
        return val.collect do |v|
          cast_dtype(v, dtype)
        end
      end

      case dtype
      when :float32
        val.to_f
      when :string
        val.to_s
      when :int32
        val.to_i
      else
        val
      end
    end

    protected

    def hashify_tensor(tensor)
      if tensor.kind_of?(Tensor) 
        tensor.to_h
      elsif tensor.kind_of?(Array)
        tensor.collect do |t| hashify_tensor(t) end
      else
        tensor
      end
    end

    def reshape(arr, shape)
      if arr.kind_of?(Array)
        return arr if shape.size < 2
        slice = shape.shift
        arr.each_slice(slice).collect do |s|
          reshape(s, shape)
        end
      else
        return arr if shape.size < 1
        slice = shape.shift
        slice.times.collect do |s|
          reshape(arr, shape.dup)
        end
      end
    end

    def auto_wrap(operand)
      if !operand.kind_of?(Tensor)
        TensorStream.constant(operand, dtype: @data_type || Tensor.detect_type(operand) )
      else
        operand
      end
    end

    def build_name
      "#{@is_const ? "Const#{Tensor.const_name}:#{@rank}" : "Variable#{Tensor.var_name}:#{@rank}"}"
    end
  end
end
