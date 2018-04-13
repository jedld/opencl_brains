module TensorStream
  class Variable < Tensor
    def initialize(data_type, rank, shape, options = {})
      @data_type = data_type
      @rank = rank
      @shape = TensorShape.new(shape, rank)
      @value = nil
      @graph = options[:graph] || TensorStream.get_default_graph
      @name = options[:name] || build_name
      if options[:initializer]
        @initalizer_tensor = options[:initializer]
      end
      @graph.add_variable(self, options)
    end

    def initializer
      @initalizer_tensor.shape = @shape
      assign(@initalizer_tensor)
    end

    def assign(value)
      Operation.new(:assign, self, value)
    end

    def assign_add(value)
      Operation.new(:assign_add, self, value)
    end

    def self.variables_initializer(collection)
      TensorStream.group(TensorStream.get_default_graph.get_collection(collection).map(&:initializer))
    end

    def self.global_variables_initializer
      variables_initializer(TensorStream::GraphKeys::GLOBAL_VARIABLES)
    end
  end
end