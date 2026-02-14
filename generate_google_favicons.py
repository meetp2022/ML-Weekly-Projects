from PIL import Image, ImageDraw

def create_favicons(root_path, assets_path):
    size_192 = 192
    size_32 = 32
    
    # helper for drawing the brand icon
    def draw_brand_icon(img_size):
        img = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 1. Background Gradient
        start_color = (102, 126, 234)
        end_color = (118, 75, 162)
        
        # Rounded mask
        mask = Image.new('L', (img_size, img_size), 0)
        mask_draw = ImageDraw.Draw(mask)
        radius = img_size // 4
        mask_draw.rounded_rectangle([0, 0, img_size-1, img_size-1], radius=radius, fill=255)
        
        # Gradient Fill
        gradient_img = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
        g_draw = ImageDraw.Draw(gradient_img)
        for x in range(img_size):
            ratio = x / img_size
            r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
            g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
            b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
            g_draw.line([(x, 0), (x, img_size)], fill=(r, g, b, 255))
            
        final_bg = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
        final_bg.paste(gradient_img, (0, 0), mask=mask)
        
        # 2. Draw Diamonds (Scaled from 32x32 original SVG coords)
        scale = img_size / 32
        d_draw = ImageDraw.Draw(final_bg)
        
        # Diamond 1
        d1 = [(16*scale, 8*scale), (20*scale, 12*scale), (16*scale, 16*scale), (12*scale, 12*scale)]
        d_draw.polygon(d1, fill=(255, 255, 255, 230))
        
        # Diamond 2
        d2 = [(16*scale, 16*scale), (20*scale, 20*scale), (16*scale, 24*scale), (12*scale, 20*scale)]
        d_draw.polygon(d2, fill=(255, 255, 255, 153))
        
        return final_bg

    # Create 192x192 PNG
    icon_192 = draw_brand_icon(size_192)
    icon_192.save(os.path.join(assets_path, 'favicon.png'))
    print(f"Created 192x192 favicon at {os.path.join(assets_path, 'favicon.png')}")
    
    # Create 32x32 ICO
    icon_32 = draw_brand_icon(size_32)
    icon_32.save(os.path.join(root_path, 'favicon.ico'), format='ICO', sizes=[(32, 32)])
    print(f"Created root-level favicon.ico at {os.path.join(root_path, 'favicon.ico')}")

if __name__ == "__main__":
    import os
    frontend_path = r'd:\Data Science\ML_weekly\ML-Weekly-Projects\ai-text-detector\AI-detector\frontend'
    assets_path = os.path.join(frontend_path, 'assets')
    create_favicons(frontend_path, assets_path)
