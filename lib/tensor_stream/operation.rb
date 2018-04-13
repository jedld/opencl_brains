module TensorStream
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank, :options

    def initialize(operation, a, b, options = {})
      @operation = operation
      @items = [a, b]
      @rank = options[:rank] || 0
      @name = set_name
      @options = options
      if options[:shape]
        @shape = TensorShape.new(options[:shape], options[:shape].size || 0)
      end
    end
    def to_s
      @name
    end

    def self.reset_counters
      @@op_counter = 0
    end
    def ruby_eval(session = Session.default_session, evaluation_cache = {})
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