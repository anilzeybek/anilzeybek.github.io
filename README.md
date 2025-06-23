# ML and Robotics Notes Blog

A Jekyll blog about machine learning and robotics algorithms.

**Live site:** [comsci.blog](https://comsci.blog)

## Local Development

### Prerequisites
- Ruby 3.0+ (recommended: install via Homebrew)
- Bundler gem

### Setup
1. Clone the repository:
   ```bash
   git clone git@github.com:anilzeybek/anilzeybek.github.io.git
   cd anilzeybek.github.io
   ```

2. Install dependencies:
   ```bash
   bundle install
   ```

3. Start the development server:
   ```bash
   bundle exec jekyll serve
   ```

4. Open your browser to [http://localhost:4000](http://localhost:4000)

### Development Notes
- The server auto-reloads when you make changes to most files
- If you edit `_config.yml`, restart the server with `Ctrl+C` then `bundle exec jekyll serve`
- New blog posts go in `_posts/` with format: `YYYY-MM-DD-title.md`

### Deployment
Push to the `main` branch - GitHub Pages will automatically build and deploy the site.

## Project Structure
- `_posts/` - Blog posts
- `_layouts/` - Page templates  
- `_includes/` - Reusable components
- `_sass/` - Stylesheets
- `assets/` - Images and other static files
- `_config.yml` - Site configuration
