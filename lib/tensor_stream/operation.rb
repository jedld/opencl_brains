module TensorStream
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank, :options

    def initialize(operation, a, b, options = {})
      @operation = operation
      @rank = options[:rank] || 0
      @name = options[:name] || set_name
      @graph = options[:graph] || TensorStream.get_default_graph
      @options = options


      @items = [a, b].map { |i| options[:preserve_params_type] ? i : auto_wrap(i) }
      @data_type = set_data_type(options[:data_type])

      if options[:shape]
        @shape = TensorShape.new(options[:shape], options[:shape].size || 0)
      end
      @graph.add_node(self)
    end

    def to_s
      @name
    end

    def self.reset_counters
      @@op_counter = 0
    end

    def to_h
      {
        op: operation,
        name: name,
        operands: hashify_tensor(items)
      }
    end

    def self.empty_matrix?(m)
      if m.kind_of?(Array)
        m.each do |item|
          if item.kind_of?(Array)
            return false if !empty_matrix?(item)
          else
            return false if item!=0 || item!=0.0
          end
        end
      end

      return true
    end

    def set_data_type(passed_data_type)
      case operation
      when :greater, :less, :equal
        :boolean
      when :shape, :rank
        :int32
      else
        passed_data_type || (@items[0] ? @items[0].data_type : :unknown)
      end
    end

    def to_math
      case operation
      when :slice
        "#{auto_math(items[0])}[#{items[1]}]"
      when :assign_sub
        "(#{items[0] ? items[0].name : "self"} -= #{auto_math(items[1])})"
      when :assign_add
        "(#{items[0] ? items[0].name : "self"} += #{auto_math(items[1])})"
      when :assign
        "(#{items[0] ? items[0].name : "self"} = #{auto_math(items[1])})"
      when :sin, :cos, :tanh
        "#{operation}(#{auto_math(items[0])})"
      when :add
       "(#{auto_math(items[0])} + #{auto_math(items[1])})"
      when :sub
        "(#{auto_math(items[0])} - #{auto_math(items[1])})"
      when :pow
        "(#{auto_math(items[0])}^#{auto_math(items[1])})"
      when :div
        "(#{auto_math(items[0])} / #{auto_math(items[1])})"
      when :mul
        if auto_math(items[0]) == 1
          auto_math(items[1])
        elsif auto_math(items[1]) == 1
          auto_math(items[0])
        else
          "(#{auto_math(items[0])} * #{auto_math(items[1])})"
        end
      when :reduce_sum
        "reduce_sum(|#{auto_math(items[0])} * #{auto_math(items[1])}|)"
      when :gradients
        "gradient(#{auto_math(items[0])})"
      when :stop_gradient
        auto_math(items[0])
      when :matmul
        "#{auto_math(items[0])}.matmul(#{auto_math(items[1])})"
      when :eye
        "eye(#{auto_math(items[0])})"
      when :transpose
        "transpose(#{auto_math(items[0])})"
      when :shape
        "#{auto_math(items[0])}.shape"
      when :exp
        "e^#{auto_math(items[0])}"
      when :ones
        "ones(#{items[0]})"
      when :flow_group
        "flow_group(#{items.collect { |i| auto_math(i)}.join(',')})"
      when :zeros
        "zeros(#{items[0]})"
      when :reshape
        "reshape(#{auto_math(items[0])},#{auto_math(items[1])})"
      when :rank
        "#{auto_math(items[0])}.rank"
      when :cond
        "(#{auto_math(options[:pred])} ? #{auto_math(items[0])} : #{auto_math(items[1])})"
      when :less
        "#{auto_math(items[0])} < #{auto_math(items[1])}"
      when :greater
        "#{auto_math(items[0])} > #{auto_math(items[1])}"
      when :square
        "#{auto_math(items[0])}\u00B2"
      when :log
        "log(#{auto_math(items[0])})"
      when :identity
        "identity(#{auto_math(items[0])})"
      when :print
        "print(#{auto_math(items[0])})"
      when :pad
        "pad(#{auto_math(items[0])},#{auto_math(options[:paddings])})"
      else
        fail "math form for #{operation}"
      end
    end

    def run
      self.eval
    end

    private

    def self.operation_counter
      @@op_counter ||= 0

      name = if @@op_counter == 0
        ""
      else
        "_#{@@op_counter}"
      end

      @@op_counter += 1
      
      name
    end

    def set_name
      "#{@operation}#{Operation.operation_counter}:#{@rank}"
    end
  end
end