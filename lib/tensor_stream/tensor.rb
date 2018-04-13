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
      @value = []
      @is_const = options[:const] || false
      @graph = options[:graph] || TensorStream.get_default_graph
      @name = options[:name] || build_name
      if options[:value]
        if options[:value].kind_of?(Array)
          @value = options[:value].collect do |v|
            TensorStream.constant(v, dtype: data_type)
          end
        else
          @value = options[:value]
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
      TensorStream::Operation.new(:add, self, operand)
    end

    def [](index)
      TensorStream::Operation.new(:slice, self, index)
    end

    def *(operand)
      TensorStream::Operation.new(:mul, self,operand)
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

    def eval
      Session.default_session.run(self)
    end

    protected

    def auto_wrap(operand)
      if !operand.kind_of?(Tensor)
        TensorStream.constant(operand)
      else
        operand
      end
    end
    def build_name
      "#{@is_const ? "Const#{Tensor.const_name}:#{@rank}" : "Variable#{Tensor.var_name}:#{@rank}"}"
    end
  end
end
