module TensorStream
  class Session
    def self.default_session
      @session ||= Session.new
    end
    def run(*args)
      options = if args.last.kind_of?(Hash)
        args.pop
      else
        {}
      end
      evaluation_cache = {}

      #scan for placeholders and assign value
      options[:feed_dict].keys.each do |k|
        if k.kind_of?(Placeholder)
          evaluation_cache[k.name.to_sym] = options[:feed_dict][k].kind_of?(Tensor) ? options[:feed_dict][k] : TensorStream.constant(options[:feed_dict][k])
        end
      end if options[:feed_dict]


      result = args.collect { |e| e.ruby_eval(self, evaluation_cache).ruby_eval(self, evaluation_cache) }

      result.size == 1 ? result.first : result
    end
  end
end