require 'json'

module TensorStream
  module Train
    class Saver
      def save(session, outputfile)
        vars = TensorStream::Graph.get_default_graph.get_collection(GraphKeys::GLOBAL_VARIABLES)
        
        variables = {}
        graph = {}
        output_dump = {
          variables: variables,
          graph: graph
        }

        vars.each do |variable|
          variables[variable.name] = variable.value
        end

        File.write(outputfile, output_dump.to_json)
      end
    end
  end
end