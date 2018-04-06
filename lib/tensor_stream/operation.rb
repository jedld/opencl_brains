module TensorStream
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank

    def initialize(operation, a, b, options = {})
      @operation = operation
      @items = [a, b]
      @rank = options[:rank] || 0
      @name = set_name
    end

    def to_s
      @name
    end

    def self.reset_counters
      @@op_counter = 0
    end

    def ruby_eval
      if operation == :add
        # ruby scalar
        if @items[0].shape.rank == 0
          TensorStream.constant(@items[0].ruby_eval + @items[1].ruby_eval, dtype: @items[0].dtype)
        end
      end
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
      @name = if @operation == :add
        "add#{Operation.operation_counter}:#{@rank}"
      elsif @operation == :slice
        "slice#{Operation.operation_counter}:#{@rank}"
      end
    end
  end
end