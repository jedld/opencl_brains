module TensorStream
  class Operation
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

    def +(operand)
      TensorStream::Operation.new(:add, self, operand)
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
      @name = if @operation = :add
        "add#{Operation.operation_counter}:#{@rank}"
      end
    end
  end
end