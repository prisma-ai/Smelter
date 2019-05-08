//
//  NSImage+Extensions.swift
//  Smelter Demo macOS
//
//  Created by Eugene Bokhan on 08/05/2019.
//  Copyright © 2019 Eugene Bokhan. All rights reserved.
//

import Cocoa

internal extension NSImage {

    var cgImage: CGImage? {
        return self.cgImage(forProposedRect: nil,
                            context: nil,
                            hints: nil)
    }

}
