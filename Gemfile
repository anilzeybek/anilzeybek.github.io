# frozen_string_literal: true

source "https://rubygems.org"
gemspec

gem "jekyll", ENV["JEKYLL_VERSION"] if ENV["JEKYLL_VERSION"]
gem "kramdown-parser-gfm" if ENV["JEKYLL_VERSION"] == "~> 3.9"

# Required for Ruby 3.4+ compatibility (local development only)
gem "csv"
gem "logger" 
gem "base64"
gem "bigdecimal"
