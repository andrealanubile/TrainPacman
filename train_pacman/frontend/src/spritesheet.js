import {Texture, Rectangle, Assets, Container, Sprite, ImageSource} from 'pixi.js'
import * as constants from './constants'

class Spritesheet {
    constructor() {
        // this.sheet = Texture.from("assets/spritesheet.png");
        this.baseTileWidth = 16;
        this.baseTileHeight = 16;
    }

    async loadSpritesheet() {
        this.sheet = await Assets.load('assets/spritesheet_transparent.png')
    }

    getImage(x, y, width, height) {
        const tileWidth = width || this.baseTileWidth;
        const tileHeight = height || this.baseTileHeight;
        const rectangle = new Rectangle(x * tileWidth, y * tileHeight, tileWidth, tileHeight);
        let texture = new Texture({
            source: this.sheet,
            frame: rectangle});

        texture.source.scaleMode = 'nearest';
        return texture;
    }

}

export default Spritesheet;