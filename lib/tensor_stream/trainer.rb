module TensorStream
  module Train
    class GradientDescentOptimizer < TensorStream::Operation
      attr_accessor :learning_rate
      
      def initialize(learning_rate, options = {})
        @items = []
        @learning_rate = learning_rate
        @operation = :gradient_descent
        @name = set_name
        @graph = options[:graph] || TensorStream.get_default_graph
        @graph.add_node(self)
      end

      def minimize(cost)
        @items << cost
        self
      end
    end
  end
end